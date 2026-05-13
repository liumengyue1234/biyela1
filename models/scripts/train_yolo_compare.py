"""
松材线虫病检测系统 - YOLO v8/v10/v11 对比训练脚本
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import time
import numpy as np


def prepare_yolo_dataset(data_root: str, output_root: str, target_size=(640, 640)):
    """
    将标注数据转换为YOLO格式

    Args:
        data_root: 原始数据根目录
        output_root: YOLO格式数据输出目录
        target_size: 目标图像尺寸
    """
    import cv2
    import numpy as np
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).parent))
    from data_preprocessing import PineNematodeDataset

    output_path = Path(output_root)

    # 创建目录
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 转换数据
    for split in ['train', 'val', 'test']:
        dataset = PineNematodeDataset(data_root, split=split)

        print(f"转换 {split} 集 ({len(dataset)} 样本)...")

        for i, (image, mask) in enumerate(tqdm(dataset)):
            # 保存图像
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
            img_resized = cv2.resize(img_np, target_size)
            img_path = output_path / 'images' / split / f"{i:06d}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

            # 调整掩码大小
            mask_resized = cv2.resize(mask.numpy().astype('uint8'), target_size,
                                      interpolation=cv2.INTER_NEAREST)

            # 查找边界框
            y_indices, x_indices = np.where(mask_resized > 0)

            if len(x_indices) > 10:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()

                # 转换为YOLO格式
                h, w = mask_resized.shape
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h

                label_path = output_path / 'labels' / split / f"{i:06d}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            else:
                label_path = output_path / 'labels' / split / f"{i:06d}.txt"
                with open(label_path, 'w') as f:
                    pass

    # 创建data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {0: 'nematode'}
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nYOLO数据集创建完成: {output_path}")
    print(f"配置文件: {yaml_path}")

    return str(yaml_path)


def train_yolo_model(data_yaml: str, model_type: str, epochs: int = 100,
                      batch_size: int = 16, img_size: int = 640, output_dir: str = None):
    """
    训练YOLO模型

    Args:
        data_yaml: 数据配置文件路径
        model_type: YOLO模型类型 (yolov8s, yolo11m, yolov10s等)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        output_dir: 输出目录
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 请安装ultralytics库")
        print("pip install ultralytics")
        return None

    print(f"\n{'='*60}")
    print(f"训练模型: {model_type}")
    print(f"参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    print(f"{'='*60}")

    # 创建或加载模型
    model = YOLO(f'{model_type}.pt')

    # 训练
    start_time = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir or 'runs/detect',
        name=f'pine_{model_type}',
        exist_ok=True,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device=0 if __import__('torch').cuda.is_available() else 'cpu'
    )
    training_time = time.time() - start_time

    # 获取验证指标
    val_results = model.val()

    metrics = {
        'model_type': model_type,
        'epochs': epochs,
        'training_time': training_time,
        'map50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
        'map50_95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
        'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
        'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
    }

    print(f"\n{model_type} 训练完成!")
    print(f"mAP@50: {metrics['map50']:.4f}")
    print(f"mAP@50:95: {metrics['map50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"训练时间: {training_time:.2f}s")

    return metrics


def compare_yolo_versions(data_yaml: str, output_dir: str, epochs: int = 100):
    """
    对比训练 YOLOv8、YOLO11、YOLOv10

    Args:
        data_yaml: 数据配置文件
        output_dir: 输出目录
        epochs: 训练轮数
    """
    # YOLO版本配置
    yolo_versions = {
        'YOLOv8s': {'model': 'yolov8s', 'params': '11.2M'},
        'YOLO11m': {'model': 'yolo11m', 'params': '25.9M'},
        'YOLOv10s': {'model': 'yolov10s', 'params': '7.2M'},
    }

    results = {}

    for version_name, config in yolo_versions.items():
        try:
            metrics = train_yolo_model(
                data_yaml=data_yaml,
                model_type=config['model'],
                epochs=epochs,
                batch_size=16,
                img_size=640,
                output_dir=output_dir
            )
            if metrics:
                metrics['params'] = config['params']
                results[version_name] = metrics
        except Exception as e:
            print(f"{version_name} 训练失败: {e}")
            results[version_name] = {'error': str(e)}

    return results


def generate_comparison_report(results: dict, output_path: str):
    """
    生成对比报告

    Args:
        results: 各模型训练结果
        output_path: 输出路径
    """
    report = "# YOLO版本对比实验报告\n\n"
    report += "## 实验配置\n\n"
    report += "- 任务：松材线虫CT图像检测\n"
    report += "- 数据集：松材线虫标注数据集（first.zip + third.zip）\n"
    report += "- 训练参数：epochs=100, batch_size=16, img_size=640\n\n"

    report += "## 实验结果\n\n"
    report += "| 模型 | 参数量 | mAP@50 | mAP@50:95 | Precision | Recall | 训练时间 |\n"
    report += "|------|--------|--------|------------|-----------|--------|----------|\n"

    for version_name, metrics in results.items():
        if 'error' in metrics:
            report += f"| {version_name} | - | - | - | - | - | 训练失败 |\n"
        else:
            report += f"| {version_name} | {metrics.get('params', '-')} | "
            report += f"{metrics.get('map50', 0):.4f} | "
            report += f"{metrics.get('map50_95', 0):.4f} | "
            report += f"{metrics.get('precision', 0):.4f} | "
            report += f"{metrics.get('recall', 0):.4f} | "
            report += f"{metrics.get('training_time', 0):.1f}s |\n"

    report += "\n## 分析结论\n\n"

    # 找出最佳模型
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_map = max(valid_results.items(), key=lambda x: x[1].get('map50_95', 0))
        best_speed = min(valid_results.items(), key=lambda x: x[1].get('training_time', float('inf')))

        report += f"1. **精度最高**: {best_map[0]} (mAP@50:95={best_map[1].get('map50_95', 0):.4f})\n"
        report += f"2. **速度最快**: {best_speed[0]} (训练时间={best_speed[1].get('training_time', 0):.1f}s)\n"
        report += f"3. **综合推荐**: 根据实际应用场景选择合适版本\n"

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n对比报告已保存: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='YOLO多版本对比训练')

    parser.add_argument('--data_root', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
                        help='原始数据根目录')
    parser.add_argument('--output', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints',
                        help='输出目录')
    parser.add_argument('--prepare_only', action='store_true',
                        help='仅准备数据集，不训练')
    parser.add_argument('--versions', type=str, default='v8,v11,v10',
                        help='要训练的版本 (v8,v11,v10)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--img_size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--compare', action='store_true', help='运行多版本对比')

    args = parser.parse_args()

    # 准备数据集
    print("=" * 60)
    print("准备YOLO数据集")
    print("=" * 60)

    data_yaml = prepare_yolo_dataset(
        data_root=args.data_root,
        output_root=str(Path(args.output) / 'yolo_compare' / 'dataset'),
        target_size=(args.img_size, args.img_size)
    )

    if args.prepare_only:
        print("\n数据集准备完成（--prepare_only 模式）")
        return

    if args.compare:
        # 运行多版本对比
        results = compare_yolo_versions(
            data_yaml=data_yaml,
            output_dir=str(Path(args.output) / 'yolo_compare'),
            epochs=args.epochs
        )

        # 生成报告
        report_path = Path(args.output) / 'yolo_compare' / 'comparison_report.md'
        generate_comparison_report(results, str(report_path))
    else:
        # 单版本训练
        version_map = {
            'v8': 'yolov8s',
            'v11': 'yolo11m',
            'v10': 'yolov10s'
        }

        for v in args.versions.split(','):
            v = v.strip()
            if v in version_map:
                train_yolo_model(
                    data_yaml=data_yaml,
                    model_type=version_map[v],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    img_size=args.img_size,
                    output_dir=str(Path(args.output) / 'yolo_compare')
                )


if __name__ == '__main__':
    main()
