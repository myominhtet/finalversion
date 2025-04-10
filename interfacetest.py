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
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
import gradio as gr
import ffmpeg

# IoU and scanner movement functions (unchanged)
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

def is_scanner_moving(prev_centroids, curr_box, scanner_id, threshold=5.0):
    x1, y1, x2, y2 = curr_box
    curr_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
    if scanner_id in prev_centroids:
        prev_x, prev_y = prev_centroids[scanner_id]
        distance = np.sqrt((curr_centroid[0] - prev_x)**2 + (curr_centroid[1] - prev_y)**2)
        return distance > threshold
    return False

def detect_video(video_path, weights, conf_thres=0.25, iou_thres=0.45, img_size=640, device='', save_dir='runs/detect/exp'):
    save_dir = Path(increment_path(Path(save_dir), exist_ok=True))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    if half:
        model.half()

    dataset = LoadImages(video_path, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    vid_path, vid_writer = None, None
    prev_centroids = {}
    scanner_id_counter = 0

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            p = Path(path)
            save_path = str(save_dir / p.name.replace('.mp4', '_output.mp4'))
            im0 = im0s

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                item_boxes, scanner_data, phone_boxes = [], [], []
                curr_scanner_boxes = []

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

                new_prev_centroids = {}
                if prev_centroids and curr_scanner_boxes:
                    for curr_box in curr_scanner_boxes:
                        curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                        best_match_id = min(prev_centroids.keys(), 
                                          key=lambda k: np.sqrt((curr_centroid[0] - prev_centroids[k][0])**2 + 
                                                                (curr_centroid[1] - prev_centroids[k][1])**2), 
                                          default=None)
                        if best_match_id is not None and np.sqrt((curr_centroid[0] - prev_centroids[best_match_id][0])**2 + 
                                                                 (curr_centroid[1] - prev_centroids[best_match_id][1])**2) < 50:
                            scanner_id = best_match_id
                        else:
                            scanner_id = scanner_id_counter
                            scanner_id_counter += 1
                        is_moving = is_scanner_moving(prev_centroids, curr_box, scanner_id)
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

                for scanner_box, movement_status, scanner_id in scanner_data:
                    x1, y1, x2, y2 = scanner_box
                    label = f"scanner {movement_status} (ID: {scanner_id})"
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[names.index("scanner")], line_thickness=2)

                product_scanning_status = ""
                payment_scanning_status = ""
                for scanner_box, movement_status, _ in scanner_data:
                    for item_box in item_boxes:
                        if movement_status == "Scanning" and compute_iou(scanner_box, item_box) > 0.1:
                            product_scanning_status = "Product scanning is finished"
                    for phone_box in phone_boxes:
                        if movement_status == "Scanning" and compute_iou(scanner_box, phone_box) > 0.1:
                            payment_scanning_status = "Payment scanning is finished"

                if product_scanning_status:
                    cv2.putText(im0, product_scanning_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)
                if payment_scanning_status:
                    cv2.putText(im0, payment_scanning_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[names.index("scanner")], 2)

            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                w, h = im0.shape[1], im0.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    # Convert to H.264 for browser compatibility
    output_h264 = str(Path(save_path).with_name(f"{Path(save_path).stem}_h264.mp4"))
    try:
        stream = ffmpeg.input(save_path)
        stream = ffmpeg.output(stream, output_h264, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        os.remove(save_path)  # Remove original
        return output_h264
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return save_path

def gradio_interface(video, conf_thres, iou_thres):
    weights = "/home/myominhtet/Desktop/deepsortfromscratch/yolov7/best.pt"
    img_size = 640
    output_video = detect_video(video, weights, conf_thres, iou_thres, img_size)
    return output_video if output_video else "Error processing video."

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

if __name__ == "__main__":
    interface.launch(share=True)