import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import ffmpeg
import gradio as gr
from fastapi import FastAPI
import uvicorn
import shutil
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

# Function to compute IoU between two boxes
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

# Function to check if a scanner is moving based on centroid displacement
def is_scanner_moving(prev_centroids, curr_box, scanner_id, threshold=2.0):
    x1, y1, x2, y2 = curr_box
    curr_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
    if scanner_id in prev_centroids:
        prev_x, prev_y = prev_centroids[scanner_id]
        distance = np.sqrt((curr_centroid[0] - prev_x)**2 + (curr_centroid[1] - prev_y)**2)
        return distance > threshold
    return False  # Default to "not moving" if no previous centroid exists

# Function to convert video to H.264 format
def convert_to_h264(input_path):
    output_path = str(Path(input_path).with_suffix('')) + "_h264.mp4"
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
        ffmpeg.run(stream, cmd='/usr/bin/ffmpeg', overwrite_output=True)
        return output_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"
        print(f"FFmpeg error: {stderr}")
        return input_path

# Detection function adapted from the second script
def detect_video(video_path, weights, conf_thres=0.25, iou_thres=0.45, img_size=640, device='', save_dir='runs/detect/exp', trace=False):
    save_dir = Path(increment_path(Path(save_dir), exist_ok=True))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    if trace:
        model = TracedModel(model, device, img_size)
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(video_path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize variables
    vid_path, vid_writer = None, None
    prev_centroids = {}
    scanner_id_counter = 0
    product_scanning_status_global = ""
    payment_scanning_status_global = ""
    old_img_b, old_img_h, old_img_w = 0, 0, 0

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
            for _ in range(3):
                model(img)[0]

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):
            p = Path(path)
            save_path = str(save_dir / p.name.replace('.mp4', '_output.mp4'))
            im0 = im0s

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                item_boxes, scanner_data, phone_boxes = [], [], []
                curr_scanner_boxes = []

                # Process each detection
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_name = names[int(cls)]
                    color = colors[int(cls)]
                    if class_name.lower() == "item":
                        item_boxes.append([x1, y1, x2, y2])
                    elif class_name.lower() == "phone":
                        phone_boxes.append([x1, y1, x2, y2])
                    elif class_name.lower() == "scanner":
                        curr_scanner_boxes.append([x1, y1, x2, y2])
                    plot_one_box(xyxy, im0, label=class_name, color=color, line_thickness=2)

                # Match scanner boxes with previous frames
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
                            if distance < 50:
                                scanner_id = best_match_id
                            else:
                                scanner_id = scanner_id_counter
                                scanner_id_counter += 1
                        else:
                            scanner_id = scanner_id_counter
                            scanner_id_counter += 1
                        is_moving = is_scanner_moving(prev_centroids, curr_box, scanner_id, threshold=2.0)
                        movement_status = "Scanning" if is_moving else "Idle"
                        scanner_data.append([curr_box, movement_status, scanner_id])
                        new_prev_centroids[scanner_id] = curr_centroid
                elif curr_scanner_boxes:
                    for curr_box in curr_scanner_boxes:
                        scanner_id = scanner_id_counter
                        scanner_id_counter += 1
                        movement_status = "Idle"
                        curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                        scanner_data.append([curr_box, movement_status, scanner_id])
                        new_prev_centroids[scanner_id] = curr_centroid

                prev_centroids = new_prev_centroids

                # Redraw scanner boxes with movement status
                for scanner_box, movement_status, scanner_id in scanner_data:
                    x1, y1, x2, y2 = scanner_box
                    label = f"scanner {movement_status} (ID: {scanner_id})"
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[names.index("scanner")], line_thickness=2)

                    # Check for overlaps only if scanning status hasn't been set
                    if not product_scanning_status_global:
                        for item_box in item_boxes:
                            iou = compute_iou(scanner_box, item_box)
                            if movement_status == "Scanning" and iou > 0.02:
                                product_scanning_status_global = "Product scanning is finished"
                                print(f"Product scanning finished at frame {i}")
                    if not payment_scanning_status_global:
                        for phone_box in phone_boxes:
                            iou = compute_iou(scanner_box, phone_box)
                            if movement_status == "Scanning" and iou > 0.02:
                                payment_scanning_status_global = "Payment scanning is finished"
                                print(f"Payment scanning finished at frame {i}")

                # Display persistent labels
                if product_scanning_status_global:
                    cv2.putText(im0, product_scanning_status_global, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)
                if payment_scanning_status_global:
                    cv2.putText(im0, payment_scanning_status_global, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)

            # Write frame to video
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                w, h = im0.shape[1], im0.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

    # Cleanup
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    # Convert to H.264
    output_h264 = str(Path(save_path).with_name(f"{Path(save_path).stem}_h264.mp4"))
    try:
        stream = ffmpeg.input(save_path)
        stream = ffmpeg.output(stream, output_h264, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
        ffmpeg.run(stream, cmd='/usr/bin/ffmpeg', overwrite_output=True)
        os.remove(save_path)
        return output_h264
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"
        print(f"FFmpeg error: {stderr}")
        return save_path

# Gradio interface
def gradio_interface(video, conf_thres, iou_thres):
    weights = "/home/myominhtet/Desktop/deepsortfromscratch/yolov7/best.pt"
    img_size = 640

    # Create a stable directory for video files
    stable_dir = "/home/myominhtet/Desktop/deepsortfromscratch/videos"
    os.makedirs(stable_dir, exist_ok=True)

    # Copy the uploaded video to a stable path
    stable_path = os.path.join(stable_dir, f"input_{Path(video).name}")
    shutil.copy(video, stable_path)
    print(f"Copied video to: {stable_path}")

    # Verify the copied file
    print(f"Stable path exists: {os.path.exists(stable_path)}")
    print(f"Stable path readable: {os.access(stable_path, os.R_OK)}")

    video = convert_to_h264(stable_path)
    output_video = detect_video(video, weights, conf_thres, iou_thres, img_size)
    
    return output_video if output_video else "Error processing video."

# Set up Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.45, step=0.05, label="IoU Threshold"),
    ],
    outputs=gr.Video(label="Processed Video"),
    title="YOLO Video Detection",
    description="Upload a video to run YOLO detection with custom parameters."
)

# Set up FastAPI app
app = FastAPI()
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)