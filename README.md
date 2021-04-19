# traffic object detection with yolov5+deeps
## Introduction
- This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5). It detects all vehicles.
- The detections of vehicles are then passed to a Deep Sort algorithm (https://github.com/ZQPei/deep_sort_pytorch) which track all vehicles.
- it's count vehicle and measure the speed of vehicle.
## requirements
`pip install -r requirements.txt`
## before run gui.py
- download the yolov5 weight from the latest realease https://github.com/ultralytics/yolov5/releases. Place the downlaoded .pt file under yolov5_model.
- download the deep sort weights from https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6. Place ckpt.t7 file under deep_sort/deep/checkpoint/.
## run 
`python gui.py`

