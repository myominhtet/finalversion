# import argparse
# import time
# from pathlib import Path
# import os
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random
# import sys
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# import numpy as np
# sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/deep_sort/")
# sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/application_util")
# from tracker import Tracker
# from nn_matching import NearestNeighborDistanceMetric
# from detection import Detection  # Import the required class or function
# from tracker import Tracker
# import visualization

# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#     save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://', 'https://'))

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size

#     if trace:
#         model = TracedModel(model, device, opt.img_size)

#     if half:
#         model.half()  # to FP16

#     # Second-stage classifier
#     classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     old_img_w = old_img_h = imgsz
#     old_img_b = 1

#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Warmup
#         if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
#             old_img_b = img.shape[0]
#             old_img_h = img.shape[2]
#             old_img_w = img.shape[3]
#             for i in range(3):
#                 model(img, augment=opt.augment)[0]

#         # Inference
#         t1 = time_synchronized()
#         with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#             pred = model(img, augment=opt.augment)[0]
#         t2 = time_synchronized()

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t3 = time_synchronized()

#         # Apply Classifier
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if webcam:  # batch_size >= 1
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             detections_list = []
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                 bbox_xywh = []
#                 confs = []
#                 classes = []

#                 for *xyxy, conf, cls in det:
#                     x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers
#                     w, h = x2 - x1, y2 - y1  # Compute width and height
#                     bbox_xywh.append([x1 + w / 2, y1 + h / 2, w, h])  # Convert to (center x, center y, width, height)
#                     confs.append(float(conf))
#                     classes.append(int(cls))

#                 # Store detections in a list for tracking
#                 detections_list.append((bbox_xywh, confs, classes))
#                 bbox_xywh, confs, _ = detections_list[0]
#                 detections = []  # List to store Detection objects

#                 for tlwh, conf in zip(bbox_xywh, confs):  # Iterate over bbox and confidence pairs
#                     detection = Detection(np.array(tlwh, dtype=np.float32), float(conf), None)  # Feature vector is None
#                     detections.append(detection)
#                     print(detections)
#                 metric = NearestNeighborDistanceMetric(
#                     metric="cosine",   # or "euclidean"
#                     matching_threshold=0.2,  # Adjust as needed
#                     budget=100  # Maximum number of features stored per track
#                 )

#                 # Now create the Tracker instance with a valid metric
#                 tracker = Tracker(metric=metric)

#                 # Call the predict function
#                 tracker.predict() 
#                 tracker.update(detections)
#                 print("Detections:")
#                 for i, det in enumerate(detections):
#                     print(f"Detection {i}: BBox: {det.tlwh}, Confidence: {det.confidence}")

#                 # Update tracker and print tracks
#                 tracker.predict()
#                 tracker.update(detections)
#                 print("\nTracked Objects:")
#                 for track in tracker.tracks:
#                     print(f"Track ID: {track.track_id}, BBox: {track.to_tlwh()}, Confirmed: {track.is_confirmed()}")
#                 print("Done!")

#                 # Draw tracked boxes
#                 if save_img or view_img:
#                     for track in tracker.tracks:
#                         bbox = track.to_tlwh()
#                         x, y, w, h = map(int, bbox)
#                         x1, y1 = x - w // 2, y - h // 2
#                         x2, y2 = x + w // 2, y + h // 2
#                         color = (0, 255, 0) if track.is_confirmed() else (0, 0, 255)
#                         cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(im0, f"ID: {track.track_id}", (x1, y1 - 10), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
#                 if view_img:
#                     cv2.imshow(str(p), im0)
#                     cv2.waitKey(0)
#                 # Save results
#                 if save_img:
#                     if dataset.mode == 'image':
#                         cv2.imwrite(save_path, im0)
#                         print(f"Image saved to: {save_path}")
#                     else:
#                         if vid_path != save_path:
#                             vid_path = save_path
#                             if isinstance(vid_writer, cv2.VideoWriter):
#                                 vid_writer.release()
#                             if vid_cap:
#                                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                             else:
#                                 fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                         vid_writer.write(im0)


#     print(f'Done. ({time.time() - t0:.3f}s)')
#     return detections_list

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     opt = parser.parse_args()
#     print(opt)
#     #check_requirements(exclude=('pycocotools', 'thop'))

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov7.pt']:
#                 detect()
#                 strip_optimizer(opt.weights)
#         else:
#  
#            detect()

#2
# import argparse
# import time
# from pathlib import Path
# import os
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random
# import sys
# import numpy as np
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/deep_sort/")
# sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/application_util")
# from tracker import Tracker
# from nn_matching import NearestNeighborDistanceMetric
# from detection import Detection
# import visualization

# def iou(bbox1, bbox2):
#     """Compute IoU between two bboxes in [x, y, w, h] format."""
#     x1, y1, w1, h1 = bbox1
#     x2, y2, w2, h2 = bbox2
#     x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
#     x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
#     x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
#     x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

#     inter_x_min = max(x1_min, x2_min)
#     inter_y_min = max(y1_min, y2_min)
#     inter_x_max = min(x1_max, x2_max)
#     inter_y_max = min(y1_max, y2_max)

#     inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
#     area1 = w1 * h1
#     area2 = w2 * h2
#     union_area = area1 + area2 - inter_area
#     return inter_area / union_area if union_area > 0 else 0

# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#     save_img = not opt.nosave and not source.endswith('.txt')
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://', 'https://'))

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
#     (save_dir / 'tracks').mkdir(parents=True, exist_ok=True)

#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'

#     # Load model
#     model = attempt_load(weights, map_location=device)
#     stride = int(model.stride.max())
#     imgsz = check_img_size(imgsz, s=stride)

#     if trace:
#         model = TracedModel(model, device, opt.img_size)
#     if half:
#         model.half()

#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Initialize Tracker with Euclidean metric and dummy features
#     metric = NearestNeighborDistanceMetric(metric="euclidean", matching_threshold=0.3, budget=100)
#     tracker = Tracker(metric=metric)

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
#     old_img_w = old_img_h = imgsz
#     old_img_b = 1

#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Warmup
#         if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
#             old_img_b = img.shape[0]
#             old_img_h = img.shape[2]
#             old_img_w = img.shape[3]
#             for i in range(3):
#                 model(img, augment=opt.augment)[0]

#         # Inference
#         t1 = time_synchronized()
#         with torch.no_grad():
#             pred = model(img, augment=opt.augment)[0]
#         t2 = time_synchronized()

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t3 = time_synchronized()

#         # Process detections
#         for i, det in enumerate(pred):
#             if webcam:
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)
#             save_path = str(save_dir / p.name)
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
#             track_txt_path = str(save_dir / 'tracks' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

#             detections_list = []
#             if len(det):
#                 # Rescale boxes
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                 bbox_xywh = []
#                 confs = []
#                 classes = []

#                 for *xyxy, conf, cls in det:
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     w, h = x2 - x1, y2 - y1
#                     bbox_xywh.append([x1 + w / 2, y1 + h / 2, w, h])
#                     confs.append(float(conf))
#                     classes.append(int(cls))

#                 # Store detections with class info
#                 detections_list.append((bbox_xywh, confs, classes))
#                 bbox_xywh, confs, classes = detections_list[0]
#                 detections = []

#                 # Create dummy features for Euclidean metric
#                 dummy_features = np.zeros((len(bbox_xywh), 128))  # Assuming feature dim is 128
#                 for tlwh, conf, cls, feat in zip(bbox_xywh, confs, classes, dummy_features):
#                     detection = Detection(np.array(tlwh, dtype=np.float32), float(conf), feat)
#                     detection.cls = cls  # Store class index in detection object
#                     detections.append(detection)

#                 # Update tracker
#                 tracker.predict()
#                 tracker.update(detections)

#                 # Print detections
#                 print("Detections:")
#                 for i, det in enumerate(detections):
#                     print(f"Detection {i}: BBox: {det.tlwh}, Confidence: {det.confidence}, Class: {names[det.cls]}")

#                 # Print and save tracked objects
#                 print("\nTracked Objects:")
#                 with open(track_txt_path + '.txt', 'a') as f:
#                     for track in tracker.tracks:
#                         bbox = track.to_tlwh()
#                         matching_det = next((d for d in detections if iou(d.tlwh, bbox) > 0.5), None)
#                         conf = matching_det.confidence if matching_det else 0.0
#                         line = f"{track.track_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {conf}\n"
#                         print(f"Track ID: {track.track_id}, BBox: {bbox}, Confirmed: {track.is_confirmed()}, Matched Conf: {conf}")
#                         f.write(line)

#                 # Draw DeepSORT tracked boxes with class names and movement status (top-left)
#                 if save_img or view_img:
#                     for track in tracker.tracks:
#                         bbox = track.to_tlwh()
#                         x, y, w, h = map(int, bbox)
#                         x1, y1 = x - w // 2, y - h // 2
#                         x2, y2 = x + w // 2, y + h // 2
#                         color = (0, 255, 0) if track.is_confirmed() else (0, 0, 255)  # Green for confirmed, red for tentative

#                         # Find matching detection to get class name
#                         matching_det = next((d for d in detections if iou(d.tlwh, bbox) > 0.5), None)
#                         class_name = names[matching_det.cls] if matching_det else "Unknown"

#                         # Create label with class name and movement status (only show status for scanners)
#                         label = f"{class_name}".strip() 
#                         cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(im0, label, (x1, y1 - 10),  # Top-left (above the box)
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # Display results
#                 if view_img:
#                     cv2.imshow(str(p), im0)
#                     key = cv2.waitKey(0) if not webcam else cv2.waitKey(1)  # Wait for key press only for images
#                     if key & 0xFF == ord('q'):
#                         break

#                 # Save results
#                 if save_img:
#                     if dataset.mode == 'image':
#                         cv2.imwrite(save_path, im0)
#                         print(f"Image saved to: {save_path}")
#                     else:
#                         if vid_path != save_path:
#                             vid_path = save_path
#                             if isinstance(vid_writer, cv2.VideoWriter):
#                                 vid_writer.release()
#                             if vid_cap:
#                                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                             else:
#                                 fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                         vid_writer.write(im0)

#     # Cleanup
#     if view_img:
#         cv2.destroyAllWindows()
#     print(f'Done. ({time.time() - t0:.3f}s)')
#     return detections_list

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     opt = parser.parse_args()
#     print(opt)

#     with torch.no_grad():
#         if opt.update:
#             for opt.weights in ['yolov7.pt']:
#                 detect()
#                 strip_optimizer(opt.weights)
#         else:
#             detect()
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
sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/deep_sort/")
sys.path.append("/home/myominhtet/Desktop/deepsortfromscratch/deep_sort/application_util")
from tracker import Tracker
from nn_matching import NearestNeighborDistanceMetric
from detection import Detection
import visualization

def iou(bbox1, bbox2):
    """Compute IoU between two bboxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_object_moving(frame_t_minus_1, frame_t, bbox, threshold=0.1):
    """Detect if an object is moving using optical flow."""
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract bbox coordinates (x, y, w, h) in tlwh format
    x, y, w, h = map(int, bbox)
    flow_bbox = flow[y:y+h, x:x+w]

    # Compute average motion magnitude
    magnitude = np.sqrt(flow_bbox[..., 0]**2 + flow_bbox[..., 1]**2)
    avg_magnitude = np.mean(magnitude)

    return avg_magnitude > threshold

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    (save_dir / 'tracks').mkdir(parents=True, exist_ok=True)

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

    # Initialize Tracker with Euclidean metric and dummy features
    metric = NearestNeighborDistanceMetric(metric="euclidean", matching_threshold=0.3, budget=100)
    tracker = Tracker(metric=metric)

    # Variables for motion detection
    prev_frame = None  # Store previous frame for optical flow

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

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
            track_txt_path = str(save_dir / 'tracks' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            detections_list = []
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh = []
                confs = []
                classes = []

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w, h = x2 - x1, y2 - y1
                    bbox_xywh.append([x1 + w / 2, y1 + h / 2, w, h])
                    confs.append(float(conf))
                    classes.append(int(cls))

                # Store detections with class info
                detections_list.append((bbox_xywh, confs, classes))
                bbox_xywh, confs, classes = detections_list[0]
                detections = []

                # Create dummy features for Euclidean metric
                dummy_features = np.zeros((len(bbox_xywh), 128))  # Assuming feature dim is 128
                for tlwh, conf, cls, feat in zip(bbox_xywh, confs, classes, dummy_features):
                    detection = Detection(np.array(tlwh, dtype=np.float32), float(conf), feat)
                    detection.cls = cls  # Store class index in detection object
                    detections.append(detection)

                # Update tracker
                tracker.predict()
                tracker.update(detections)

                # Print detections
                print("Detections:")
                for i, det in enumerate(detections):
                    print(f"Detection {i}: BBox: {det.tlwh}, Confidence: {det.confidence}, Class: {names[det.cls]}")

                # Print and save tracked objects
                print("\nTracked Objects:")
                with open(track_txt_path + '.txt', 'a') as f:
                    for track in tracker.tracks:
                        bbox = track.to_tlwh()
                        matching_det = next((d for d in detections if iou(d.tlwh, bbox) > 0.5), None)
                        conf = matching_det.confidence if matching_det else 0.0
                        line = f"{track.track_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {conf}\n"
                        print(f"Track ID: {track.track_id}, BBox: {bbox}, Confirmed: {track.is_confirmed()}, Matched Conf: {conf}")
                        f.write(line)

                # Draw DeepSORT tracked boxes with class names and movement status (top-left)
                if save_img or view_img:
                    for track in tracker.tracks:
                        if not track.is_confirmed():
                            continue
                        bbox = track.to_tlwh()
                        x, y, w, h = map(int, bbox)
                        x1, y1 = x - w // 2, y - h // 2
                        x2, y2 = x + w // 2, y + h // 2
                        color = (0, 255, 0) if track.is_confirmed() else (0, 0, 255)  # Green for confirmed, red for tentative

                        # Find matching detection to get class name
                        matching_det = next((d for d in detections if iou(d.tlwh, bbox) > 0.3), None)
                        class_name = names[matching_det.cls] if matching_det else "Unknown"

                        # Check motion for "scanner" class only
                        movement_status = ""
                        if class_name.lower() == "scanner" and prev_frame is not None:
                            # Ensure bbox stays within frame bounds
                            x, y, w, h = max(0, x), max(0, y), min(w, im0.shape[1] - x), min(h, im0.shape[0] - y)
                            if w > 0 and h > 0:  # Valid bbox
                                is_moving = is_object_moving(prev_frame, im0, [x, y, w, h], threshold=1.0)
                                movement_status = "Scanning" if is_moving else " "

                        # Create label with class name and movement status
                        label = f"{class_name} {movement_status}".strip()
                        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(im0, label, (x1, y1 - 10),  # Top-left (above the box)
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Display results
                if view_img:
                    cv2.imshow(str(p), im0)
                    key = cv2.waitKey(0) if not webcam else cv2.waitKey(1)  # Wait for key press only for images
                    if key & 0xFF == ord('q'):
                        break

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

            # Update previous frame
            prev_frame = im0.copy()

    # Cleanup
    if view_img:
        cv2.destroyAllWindows()
    print(f'Done. ({time.time() - t0:.3f}s)')
    return detections_list

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