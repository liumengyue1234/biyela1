"""
松材线虫病检测系统 - YOLO训练脚本
"""

import sys
import argparse
from pathlib import Path
import yaml


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
    import shutil
    
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
            
            if len(x_indices) > 10:  # 确保有足够的像素
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                
                # 转换为YOLO格式（归一化中心点+宽高）
                h, w = mask_resized.shape
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                
                # 保存标注
                label_path = output_path / 'labels' / split / f"{i:06d}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            else:
                # 空标注
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


def train_yolo(data_yaml: str, model_type: str = 'yolov8s', epochs: int = 100, 
               batch_size: int = 16, img_size: int = 640, output_dir: str = None):
    """
    训练YOLO模型
    
    Args:
        data_yaml: 数据配置文件路径
        model_type: YOLO模型类型
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
    
    print(f"使用YOLO模型: {model_type}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    
    # 创建或加载模型
    model = YOLO(f'{model_type}.pt')
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir or 'runs/detect',
        name='pine_nematode',
        exist_ok=True,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device=0 if __import__('torch').cuda.is_available() else 'cpu'
    )
    
    print("\n训练完成!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='训练YOLO模型')
    
    parser.add_argument('--data_root', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
                        help='原始数据根目录')
    parser.add_argument('--output', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints/yolo',
                        help='输出目录')
    parser.add_argument('--prepare_only', action='store_true',
                        help='仅准备数据集，不训练')
    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLO模型类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--img_size', type=int, default=640, help='图像尺寸')
    
    args = parser.parse_args()
    
    # 准备数据集
    print("=" * 60)
    print("准备YOLO数据集")
    print("=" * 60)
    
    data_yaml = prepare_yolo_dataset(
        data_root=args.data_root,
        output_root=str(Path(args.output) / 'dataset'),
        target_size=(args.img_size, args.img_size)
    )
    
    if args.prepare_only:
        print("\n数据集准备完成（--prepare_only 模式）")
        return
    
    # 训练模型
    print("\n" + "=" * 60)
    print("训练YOLO模型")
    print("=" * 60)
    
    train_yolo(
        data_yaml=data_yaml,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
