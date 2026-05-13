"""
松材线虫病检测系统 - YOLO模型
使用Ultralytics YOLOv8进行目标检测
"""

import sys
import argparse
from pathlib import Path
import torch
import yaml


class YOLOModel:
    """YOLO模型包装类"""
    
    def __init__(
        self,
        model_type: str = 'yolov8n',
        num_classes: int = 1,
        pretrained: bool = True
    ):
        """
        初始化YOLO模型
        
        Args:
            model_type: YOLO模型类型 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            num_classes: 类别数
            pretrained: 是否使用预训练权重
        """
        self.model_type = model_type
        self.num_classes = num_classes
        
        try:
            from ultralytics import YOLO
            self.uses_ultralytics = True
        except ImportError:
            self.uses_ultralytics = False
            print("Warning: ultralytics未安装，使用模拟YOLO模型")
    
    def create_model(self):
        """创建YOLO模型"""
        if self.uses_ultralytics:
            # 使用ultralytics库
            model = YOLO(f'{self.model_type}.pt' if self.num_classes == 80 else 
                        f'{self.model_type}.yaml')
            
            # 修改输出层以适应自定义类别数
            if self.num_classes != 80:
                # 重新定义模型
                pass
            
            return model
        else:
            # 使用简化版模型（用于演示）
            return self._create_simple_model()
    
    def _create_simple_model(self):
        """创建简化版YOLO模型"""
        import torch.nn as nn
        
        class SimpleYOLO(nn.Module):
            def __init__(self, num_classes=1):
                super().__init__()
                # 简化的backbone
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.SiLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.SiLU(),
                    
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.SiLU(),
                )
                
                # 检测头
                self.head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.SiLU(),
                    nn.Conv2d(128, (5 + num_classes) * 3, 1)  # 3个锚框
                )
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x
        
        return SimpleYOLO(self.num_classes)


class YOLOTrainer:
    """YOLO训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, data_yaml: str, epochs: int = 100):
        """训练YOLO模型"""
        if self.config.get('use_ultralytics', False):
            from ultralytics import YOLO
            
            model = YOLO(f"{self.config.get('model_type', 'yolov8s')}.pt")
            
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=self.config.get('img_size', 640),
                batch=self.config.get('batch_size', 16),
                device=self.device,
                project=self.config.get('project', 'runs/detect'),
                name=self.config.get('name', 'train'),
                exist_ok=True,
                optimizer=self.config.get('optimizer', 'SGD'),
                lr0=self.config.get('learning_rate', 0.01),
                patience=self.config.get('patience', 50),
                save=True,
                save_period=10,
                val=True,
                plots=True
            )
            
            return results
        else:
            print("请安装ultralytics库以使用完整训练功能:")
            print("pip install ultralytics")
            return None
    
    def export_model(self, weights_path: str, format: str = 'onnx'):
        """导出模型"""
        if self.config.get('use_ultralytics', False):
            from ultralytics import YOLO
            
            model = YOLO(weights_path)
            model.export(format=format)
        else:
            print("导出功能需要ultralytics库")


def create_yolo_dataset(data_root: str, output_root: str):
    """
    将标注数据转换为YOLO格式
    
    Args:
        data_root: 原始数据根目录
        output_root: YOLO格式数据输出目录
    """
    from data_preprocessing import PineNematodeDataset
    import cv2
    import json
    import shutil
    from tqdm import tqdm
    
    output_path = Path(output_root)
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 转换数据
    for split in ['train', 'val', 'test']:
        dataset = PineNematodeDataset(data_root, split=split)
        
        print(f"转换 {split} 集...")
        for i, (image, mask) in enumerate(tqdm(dataset)):
            # 保存图像
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
            img_path = output_path / 'images' / split / f"{i:06d}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # 保存标注（YOLO格式）
            # 找到mask中的目标区域并计算边界框
            mask_np = mask.numpy()
            y_indices, x_indices = np.where(mask_np > 0)
            
            if len(x_indices) > 0:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                
                # 转换为YOLO格式（归一化中心点+宽高）
                h, w = mask_np.shape
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                
                label_path = output_path / 'labels' / split / f"{i:06d}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
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
    
    print(f"YOLO数据集创建完成: {output_path}")
    print(f"配置文件: {yaml_path}")
    
    return str(yaml_path)


def create_yolo_config():
    """创建YOLO配置文件"""
    config = {
        # 模型配置
        'model_type': 'yolov8s',  # n/s/m/l/x
        'img_size': 640,
        'num_classes': 1,
        
        # 训练配置
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'patience': 50,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # 输出
        'project': 'D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints/yolo',
        'name': 'train'
    }
    
    return config


if __name__ == '__main__':
    # 测试YOLO模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = YOLOModel(model_type='yolov8n', num_classes=1)
    print("YOLO模型包装类创建成功")
