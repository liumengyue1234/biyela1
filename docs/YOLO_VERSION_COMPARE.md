# YOLO多版本对比说明

## 概述

本系统集成了三种YOLO版本进行对比实验：

| 版本 | 机构 | 发布年份 | 特点 |
|------|------|----------|------|
| YOLOv8 | Ultralytics | 2023 | 经典版本，精度与速度平衡 |
| YOLO11 | Ultralytics | 2024 | 新版，增强特征提取能力 |
| YOLOv10 | 清华大学 | 2024 | 端到端设计，无需NMS后处理 |

## 训练对比

### 1. 单独训练某个版本

```bash
# 训练 YOLOv8
python models/scripts/train_yolo_compare.py --versions v8 --epochs 100

# 训练 YOLO11
python models/scripts/train_yolo_compare.py --versions v11 --epochs 100

# 训练 YOLOv10
python models/scripts/train_yolo_compare.py --versions v10 --epochs 100
```

### 2. 运行三版本对比训练

```bash
python models/scripts/train_yolo_compare.py --compare --epochs 100
```

### 3. 仅准备数据集

```bash
python models/scripts/train_yolo_compare.py --prepare_only
```

## 检测对比

### 1. 对比检测单张图像

```bash
python models/scripts/detect_yolo_compare.py --image path/to/image.jpg --visualize
```

### 2. 指定版本检测

```bash
# 检测 YOLOv8
python models/scripts/detect_yolo_compare.py --image path/to/image.jpg --versions v8

# 检测 YOLO11
python models/scripts/detect_yolo_compare.py --image path/to/image.jpg --versions v11

# 检测 YOLOv10
python models/scripts/detect_yolo_compare.py --image path/to/image.jpg --versions v10
```

### 3. 参数说明

- `--image, -i`: 测试图像路径（必需）
- `--checkpoint_dir, -c`: 模型权重目录（默认: checkpoints）
- `--output, -o`: 输出目录（默认: outputs）
- `--versions`: 要对比的版本 (all/v8/v11/v10)
- `--conf`: 置信度阈值（默认: 0.25）
- `--visualize`: 生成可视化对比图

## 依赖安装

确保已安装ultralytics库：

```bash
pip install ultralytics
```

## 预期性能

| 模型 | mAP@50 | mAP@50:95 | 参数量 | 推理速度 |
|------|--------|-----------|--------|----------|
| YOLOv8s | 85.8% | 76.5% | 11.2M | 快 |
| YOLO11m | 86.5% | 78.2% | 25.9M | 快 |
| YOLOv10s | 84.2% | 75.8% | 7.2M | 最快 |

## 注意事项

1. YOLOv10需要安装清华大学的官方实现：
   ```bash
   pip install yolo10
   ```

2. 首次运行会自动下载预训练权重

3. 建议使用GPU进行训练以加快速度

4. 训练输出保存在 `models/checkpoints/yolo_compare/` 目录
