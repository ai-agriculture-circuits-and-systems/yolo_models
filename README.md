# YOLO Models

YOLO (You Only Look Once) is a family of object detection models that are known for their speed and accuracy. These models are widely used in various applications such as autonomous driving, security, and more.

## Installation
To download weights, use the helper script (recommended) or the `wget` commands below:

```bash
./scripts/download_yolo_models.sh              # all README weights (+ URL overrides)
./scripts/download_yolo_models.sh --regression-only   # minimal set for regression tests
./scripts/download_yolo_models.sh --retry-failed --force   # re-fetch previously 404 weights
```

Legacy YOLOv1/YOLOv2 `.weights` on pjreddie.com are mostly offline; `download_yolo_models.sh` uses mirrors from `scripts/yolo_model_url_overrides.tsv` (except `yolov1-tiny.weights`, which has no working public file).

## [YOLOv1 Models](https://github.com/pjh5672/YOLOv1) (Original YOLO by Joseph Redmon)
- Original YOLOv1: wget -P models https://pjreddie.com/media/files/yolov1/yolov1.weights
- YOLOv1 Tiny: *(no stable mirror; use YOLOv1 full weights above or YOLOv3+)*

## [YOLOv2 Models](https://github.com/pjh5672/YOLOv2) (YOLO9000 by Joseph Redmon)
- YOLOv2: wget -P models https://pjreddie.com/media/files/yolov2.weights
- YOLOv2 Tiny: wget -P models https://pjreddie.com/media/files/yolov2-tiny.weights

## [YOLOv3 Models](https://github.com/ultralytics/yolov3) (by Ultralytics)
- YOLOv3: wget -P models https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3.pt
- YOLOv3-SPP: wget -P models https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3-spp.pt
- YOLOv3-Tiny: wget -P models https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3-tiny.pt

## [YOLOv4 Models](https://github.com/AlexeyAB/darknet) (by Alexey Bochkovskiy)
- YOLOv4: wget -P models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
- YOLOv4 Tiny: wget -P models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
- YOLOv4-CSP: wget -P models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights
- YOLOv4x-mish: wget -P models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights

## [YOLOv5 Models](https://github.com/ultralytics/yolov5) (by Ultralytics)
### Detection Models
- YOLOv5 Nano: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
- YOLOv5 Small: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
- YOLOv5 Medium: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
- YOLOv5 Large: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
- YOLOv5 XLarge: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt

### Segmentation Models
- YOLOv5 Nano Segmentation: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt
- YOLOv5 Small Segmentation: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt
- YOLOv5 Medium Segmentation: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt
- YOLOv5 Large Segmentation: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt
- YOLOv5 XLarge Segmentation: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt

### Classification Models
- YOLOv5 Nano Classification: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt
- YOLOv5 Small Classification: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt
- YOLOv5 Medium Classification: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt
- YOLOv5 Large Classification: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt
- YOLOv5 XLarge Classification: wget -P models https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-cls.pt

## [YOLOv6 Models](https://github.com/meituan/YOLOv6) (by Meituan)
### Detection Models
- YOLOv6 Nano: wget -P models https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt
- YOLOv6 Small: wget -P models https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
- YOLOv6 Medium: wget -P models https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt
- YOLOv6 Large: wget -P models https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt

## [YOLOv7 Models](https://github.com/WongKinYiu/yolov7) (by WongKinYiu)
### Detection Models
- YOLOv7: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
- YOLOv7 Tiny: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
- YOLOv7 W6: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
- YOLOv7 E6: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
- YOLOv7 D6: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
- YOLOv7 E6E: wget -P models https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

## [YOLOv8 Models](https://github.com/ultralytics/) (by Ultralytics)
### Detection Models
- YOLOv8 Nano: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- YOLOv8 Small: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
- YOLOv8 Medium: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
- YOLOv8 Large: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
- YOLOv8 XLarge: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

### Instance Segmentation Models
- YOLOv8 Nano Segmentation: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
- YOLOv8 Small Segmentation: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt
- YOLOv8 Medium Segmentation: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt
- YOLOv8 Large Segmentation: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt
- YOLOv8 XLarge Segmentation: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt

### Pose Estimation Models
- YOLOv8 Nano Pose: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
- YOLOv8 Small Pose: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
- YOLOv8 Medium Pose: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt
- YOLOv8 Large Pose: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt
- YOLOv8 XLarge Pose: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt

### Classification Models
- YOLOv8 Nano Classification: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt
- YOLOv8 Small Classification: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt
- YOLOv8 Medium Classification: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt
- YOLOv8 Large Classification: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt
- YOLOv8 XLarge Classification: wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt

## [YOLOv9 Models](https://github.com/WongKinYiu/yolov9) (by WongKinYiu)
### Detection Models
- YOLOv9-C: wget -P models https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
- YOLOv9-E: wget -P models https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
- YOLOv9-S: wget -P models https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt

## [YOLOv10 Models](https://github.com/THU-MIG/yolov10) (by THU-MIG / IDEA-Research)
### Detection Models
- YOLOv10 Nano: wget -P models https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
- YOLOv10 Small: wget -P models https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt
- YOLOv10 Medium: wget -P models https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt
- YOLOv10 Large: wget -P models https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt
- YOLOv10 XLarge: wget -P models https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt

## [YOLO11 Models](https://github.com/ultralytics) (by Ultralytics)
- YOLO11 Nano: wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
- YOLO11 Small: wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
- YOLO11 Medium: wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
- YOLO11 Large: wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
- YOLO11 XLarge: wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

## Training (mini VOC or custom YAML)

Prepare data and weights, then train any checkpoint under `models/` by stem name:

```bash
./scripts/download_yolo_models.sh --sync-sources --install-deps
./scripts/create_mini_voc.sh
./scripts/prepare_mini_voc_yolo.py   # or via regression_test.sh

./scripts/yolo.sh models --trainable-only    # list trainable checkpoints
./scripts/yolo.sh train --model yolov5n --epochs 100 --device 0
./scripts/yolo.sh train --model yolov10n --epochs 1 --device cpu
./scripts/regression_test.sh                 # 8-model smoke test (one per backend)
./scripts/regression_test.sh --all-trainable # all checkpoints in models/ on mini VOC
```

Legacy Darknet `.weights` (YOLOv1/v2/v4) train via AlexeyAB `darknet detector train`.
Source is a git submodule at `tools/darknet-src` ([AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)).

```bash
git submodule update --init tools/darknet-src   # if you cloned without --recurse-submodules
./scripts/fetch_darknet_cfg.sh
./scripts/install_darknet.sh --cpu              # build tools/darknet/darknet (use GPU build without --cpu)
./scripts/yolo.sh train --model yolov4-tiny --epochs 1 --device cpu
```

Inference still uses `yolo/detector.py`. `yolov1-tiny.weights` has no public mirror (see `scripts/yolo_model_unavailable.txt`).

## Usage
For detailed usage instructions, please refer to the [USAGE.md](USAGE.md) file.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please contact liyongfu.sg@gmail.com.
