import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Function to compute IoU between two boxes
def compute_iou(box1, box2):
    # box1 and box2 are in [x1, y1, x2, y2] format
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate areas of both boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Function to check if a scanner is moving based on centroid displacement
def is_scanner_moving(prev_centroids, curr_box, scanner_id, threshold=2.0):
    """Determine if a scanner is moving based on centroid displacement."""
    x1, y1, x2, y2 = curr_box
    curr_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)  # Center of the bbox (x, y)

    if scanner_id in prev_centroids:
        prev_x, prev_y = prev_centroids[scanner_id]
        distance = np.sqrt((curr_centroid[0] - prev_x)**2 + (curr_centroid[1] - prev_y)**2)
        return distance > threshold
    return False  # Default to "not moving" if no previous centroid exists

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Dictionary to store previous centroids for scanners (using a simple counter as ID)
    prev_centroids = {}
    scanner_id_counter = 0  # Simple ID for each new scanner detected

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Initialize lists to store bounding boxes and scanner data
                item_boxes = []
                scanner_data = []  # Store [box, movement_status, scanner_id]
                phone_boxes = []

                # Temporary storage for scanner boxes to match with previous frames
                curr_scanner_boxes = []

                # Process each detection
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w, h = x2 - x1, y2 - y1
                    class_name = names[int(cls)]
                    color = colors[int(cls)]

                    # Default movement_status for all classes
                    movement_status = ""

                    if class_name.lower() == "item":
                        item_boxes.append([x1, y1, x2, y2])
                    elif class_name.lower() == "phone":
                        phone_boxes.append([x1, y1, x2, y2])
                    elif class_name.lower() == "scanner":
                        curr_scanner_boxes.append([x1, y1, x2, y2])

                    # Draw bounding box and label (will update scanner labels later)
                    label = f"{class_name} {movement_status}".strip()
                    plot_one_box(xyxy, im0, label=label, color=color, line_thickness=2)

                # Match current scanner boxes with previous ones based on proximity
                new_prev_centroids = {}
                if prev_centroids and curr_scanner_boxes:
                    for curr_box in curr_scanner_boxes:
                        curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                        best_match_id = min(prev_centroids.keys(), 
                                          key=lambda k: np.sqrt((curr_centroid[0] - prev_centroids[k][0])**2 + 
                                                                (curr_centroid[1] - prev_centroids[k][1])**2), 
                                          default=None)
                        if best_match_id is not None:
                            distance = np.sqrt((curr_centroid[0] - prev_centroids[best_match_id][0])**2 + 
                                              (curr_centroid[1] - prev_centroids[best_match_id][1])**2)
                            if distance < 50:  # Threshold to consider it the same scanner
                                scanner_id = best_match_id
                            else:
                                scanner_id = scanner_id_counter
                                scanner_id_counter += 1
                        else:
                            scanner_id = scanner_id_counter
                            scanner_id_counter += 1

                        # Check if moving
                        is_moving = is_scanner_moving(prev_centroids, curr_box, scanner_id, threshold=2.0)
                        movement_status = "Scanning" if is_moving else "Idle"
                        scanner_data.append([curr_box, movement_status, scanner_id])
                        new_prev_centroids[scanner_id] = curr_centroid
                elif curr_scanner_boxes:  # First frame with scanners
                    for curr_box in curr_scanner_boxes:
                        scanner_id = scanner_id_counter
                        scanner_id_counter += 1
                        movement_status = "Idle"  # No movement info in first frame
                        curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                        scanner_data.append([curr_box, movement_status, scanner_id])
                        new_prev_centroids[scanner_id] = curr_centroid

                # Update prev_centroids
                prev_centroids = new_prev_centroids

                # Redraw scanner boxes with movement status
                for scanner_box, movement_status, scanner_id in scanner_data:
                    x1, y1, x2, y2 = scanner_box
                    color = colors[names.index("scanner")]
                    label = f"scanner {movement_status} (ID: {scanner_id})"
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=color, line_thickness=2)

                # Compute IoU and display scanning status
                product_scanning_status = ""
                payment_scanning_status = ""
                for scanner_box, movement_status, scanner_id in scanner_data:
                    for item_box in item_boxes:
                        iou = compute_iou(scanner_box, item_box)
                        print(f"IoU between scanner {scanner_box} and item {item_box}: {iou:.4f}")
                        if movement_status == "Scanning" and iou > 0.001:  # Adjusted IoU threshold
                            print("Product scanning is finished")
                            product_scanning_status = "Product scanning is finished"
                    for phone_box in phone_boxes:
                        iou = compute_iou(scanner_box, phone_box)
                        print(f"IoU between scanner {scanner_box} and phone {phone_box}: {iou:.4f}")
                        if movement_status == "Scanning" and iou > 0.001:  # Adjusted IoU threshold
                            print("Payment scanning is finished")
                            payment_scanning_status = "Payment scanning is finished"

                # Display scanning status
                if product_scanning_status:
                    cv2.putText(im0, product_scanning_status, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)
                if payment_scanning_status:
                    cv2.putText(im0, payment_scanning_status, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)

                # Save results
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f"Image saved to: {save_path}")
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

            # Display results
            if view_img:
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1) if webcam else cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break

    # Cleanup
    if view_img:
        cv2.destroyAllWindows()
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()