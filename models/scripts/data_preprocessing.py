"""
松材线虫病检测系统 - 数据预处理模块
处理Labelme格式的标注数据，转换为模型可用的格式
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PineNematodeDataset(Dataset):
    """松材线虫数据集类"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分 'train', 'val', 'test'
            transform: 数据增强
            target_size: 目标图像尺寸
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # 收集所有标注文件
        self.samples = self._collect_samples()
        
        # 划分数据集
        if split in ['train', 'val', 'test']:
            self._split_dataset()
    
    def _collect_samples(self) -> List[Dict]:
        """收集所有样本"""
        samples = []
        
        # 遍历所有scan目录
        for scan_dir in self.data_root.glob('scan*'):
            if scan_dir.is_dir():
                # 查找对应的标注文件
                for json_file in scan_dir.glob('*.json'):
                    # 获取对应的图像文件
                    image_name = json_file.stem + '.png'
                    image_path = json_file.parent / image_name
                    
                    if image_path.exists():
                        samples.append({
                            'image_path': str(image_path),
                            'json_path': str(json_file),
                            'scan_dir': str(scan_dir.name)
                        })
        
        return samples
    
    def _split_dataset(self):
        """划分训练集、验证集、测试集"""
        train_samples, temp_samples = train_test_split(
            self.samples, test_size=0.2, random_state=42
        )
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42
        )
        
        split_map = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        self.samples = split_map.get(self.split, [])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取样本"""
        sample = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标注
        mask = self._load_mask(sample['json_path'], image.shape[:2])
        
        # 调整尺寸
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 数据增强
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _load_mask(self, json_path: str, shape: Tuple[int, int]) -> np.ndarray:
        """
        从Labelme JSON文件加载掩码
        
        Args:
            json_path: JSON文件路径
            shape: 掩码形状 (H, W)
        
        Returns:
            掩码数组，形状为 (H, W)，值为类别ID
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 类别映射
        class_map = {
            'Vector insect': 1,  # 松材线虫
            'nematode': 1,
            'insect': 1,
            'background': 0
        }
        
        # 绘制每个标注形状
        for shape_data in data.get('shapes', []):
            label = shape_data.get('label', '').lower()
            points = shape_data.get('points', [])
            shape_type = shape_data.get('shape_type', '')
            
            # 跳过无效标签
            if label in ['hsdoiaihod', 'background'] or not points:
                continue
            
            class_id = class_map.get(label, 1)
            
            # 绘制形状
            if shape_type == 'rectangle':
                pts = np.array(points, dtype=np.int32)
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), class_id, -1)
            
            elif shape_type == 'polygon':
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], class_id)
            
            elif shape_type == 'circle':
                if len(points) >= 2:
                    center = tuple(map(int, points[0]))
                    radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[1])))
                    cv2.circle(mask, center, radius, class_id, -1)
        
        return mask


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(
        self,
        data_root: str,
        output_root: str,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        初始化预处理器
        
        Args:
            data_root: 原始数据根目录
            output_root: 处理后数据输出目录
            target_size: 目标图像尺寸
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.target_size = target_size
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 类别定义
        self.classes = ['background', 'nematode']
        self.num_classes = len(self.classes)
    
    def process_dataset(self, dataset_name: str) -> Dict[str, List[str]]:
        """
        处理数据集
        
        Args:
            dataset_name: 数据集名称 (first, third)
        
        Returns:
            划分后的数据集路径信息
        """
        dataset_path = self.data_root / dataset_name
        output_path = self.output_root / dataset_name
        
        print(f"处理数据集: {dataset_name}")
        print(f"原始路径: {dataset_path}")
        print(f"输出路径: {output_path}")
        
        # 创建子目录
        splits = ['train', 'val', 'test']
        for split in splits:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
        
        # 收集所有样本
        samples = self._collect_samples(dataset_path)
        print(f"找到 {len(samples)} 个样本")
        
        # 划分数据集
        train_samples, temp_samples = train_test_split(samples, test_size=0.2, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
        
        splits_data = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        # 处理并保存每个划分
        for split, samples in splits_data.items():
            print(f"处理 {split} 集 ({len(samples)} 样本)...")
            for i, sample in enumerate(samples):
                self._process_sample(sample, output_path, split)
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(samples)}")
        
        print(f"数据集处理完成: {output_path}")
        
        return {
            'root': str(output_path),
            'train_size': len(train_samples),
            'val_size': len(val_samples),
            'test_size': len(test_samples)
        }
    
    def _collect_samples(self, dataset_path: Path) -> List[Dict]:
        """收集所有样本"""
        samples = []
        
        for scan_dir in dataset_path.glob('scan*'):
            if scan_dir.is_dir():
                for json_file in scan_dir.glob('*.json'):
                    image_name = json_file.stem + '.png'
                    image_path = json_file.parent / image_name
                    
                    if image_path.exists():
                        samples.append({
                            'image_path': str(image_path),
                            'json_path': str(json_file)
                        })
        
        return samples
    
    def _process_sample(self, sample: Dict, output_path: Path, split: str):
        """处理单个样本"""
        # 读取图像
        image = cv2.imread(sample['image_path'])
        if image is None:
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # 读取并绘制掩码
        mask = self._load_mask(sample['json_path'], original_shape)
        
        # 调整尺寸
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 保存
        base_name = Path(sample['image_path']).stem
        
        image_path = output_path / 'images' / split / f"{base_name}.png"
        mask_path = output_path / 'masks' / split / f"{base_name}.png"
        
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)
    
    def _load_mask(self, json_path: str, shape: Tuple[int, int]) -> np.ndarray:
        """从Labelme JSON加载掩码"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        class_map = {
            'Vector insect': 1,
            'nematode': 1,
            'insect': 1
        }
        
        for shape_data in data.get('shapes', []):
            label = shape_data.get('label', '').lower()
            points = shape_data.get('points', [])
            shape_type = shape_data.get('shape_type', '')
            
            if label in ['hsdoiaihod', 'background'] or not points:
                continue
            
            class_id = class_map.get(label, 1)
            
            if shape_type == 'rectangle':
                pts = np.array(points, dtype=np.int32)
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), class_id, -1)
            
            elif shape_type == 'polygon':
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], class_id)
            
            elif shape_type == 'circle':
                if len(points) >= 2:
                    center = tuple(map(int, points[0]))
                    radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[1])))
                    cv2.circle(mask, center, radius, class_id, -1)
        
        return mask


def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512)
) -> Dict[str, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_root: 处理后数据根目录
        batch_size: 批大小
        num_workers: 工作进程数
        target_size: 目标尺寸
    
    Returns:
        包含train, val, test数据加载器的字典
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = PineNematodeDataset(
            data_root=data_root,
            split=split,
            target_size=target_size
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return dataloaders


if __name__ == '__main__':
    # 测试数据预处理
    preprocessor = DataPreprocessor(
        data_root='D:/松材线虫/标注',
        output_root='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
        target_size=(512, 512)
    )
    
    # 处理first数据集
    first_stats = preprocessor.process_dataset('first')
    print(f"First数据集: {first_stats}")
    
    # 处理third数据集
    third_stats = preprocessor.process_dataset('third')
    print(f"Third数据集: {third_stats}")
