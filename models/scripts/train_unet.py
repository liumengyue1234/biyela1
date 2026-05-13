"""
松材线虫病检测系统 - U-Net训练脚本
支持多数据集训练，完整的训练流程
"""

import os
import sys
import argparse
import time
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入自定义模块
sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessing import PineNematodeDataset, DataPreprocessor
from unet_model import ImprovedUNet, CombinedLoss, DiceLoss, create_improved_unet


class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.dice_scores = []
        self.ious = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
    
    def update(self, loss, pred, target, num_classes=2):
        self.losses.append(loss.item())
        
        # 计算Dice Score
        pred_idx = pred.argmax(dim=1)
        dice = self._compute_dice(pred_idx, target, num_classes)
        self.dice_scores.append(dice)
        
        # 计算IoU
        iou = self._compute_iou(pred_idx, target, num_classes)
        self.ious.append(iou)
        
        # 计算Precision, Recall, F1
        precision, recall, f1 = self._compute_prf(pred_idx, target, num_classes)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
    
    def _compute_dice(self, pred, target, num_classes):
        """计算Dice系数"""
        dice_scores = []
        for c in range(1, num_classes):  # 跳过背景类
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            if union > 0:
                dice = (2. * intersection / union).item()
                dice_scores.append(dice)
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def _compute_iou(self, pred, target, num_classes):
        """计算IoU"""
        ious = []
        for c in range(1, num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            if union > 0:
                iou = (intersection / union).item()
                ious.append(iou)
        return np.mean(ious) if ious else 0.0
    
    def _compute_prf(self, pred, target, num_classes):
        """计算精确率、召回率、F1"""
        # 只计算前景类
        pred_pos = (pred == 1).float()
        target_pos = (target == 1).float()
        
        tp = (pred_pos * target_pos).sum().item()
        fp = (pred_pos * (1 - target_pos)).sum().item()
        fn = ((1 - pred_pos) * target_pos).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def get_avg_metrics(self):
        return {
            'loss': np.mean(self.losses) if self.losses else 0,
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0,
            'iou': np.mean(self.ious) if self.ious else 0,
            'precision': np.mean(self.precisions) if self.precisions else 0,
            'recall': np.mean(self.recalls) if self.recalls else 0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0
        }


class Trainer:
    """U-Net训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        # 训练历史
        self.history = {
            'train': [],
            'val': []
        }
        
        # 最佳模型
        self.best_dice = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        tracker = MetricsTracker()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录指标
            running_loss += loss.item()
            tracker.update(loss, outputs, masks)
            
            # 更新进度条
            metrics = tracker.get_avg_metrics()
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'iou': f"{metrics['iou']:.4f}"
            })
        
        avg_metrics = tracker.get_avg_metrics()
        avg_metrics['loss'] = running_loss / len(self.train_loader)
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """验证"""
        self.model.eval()
        
        tracker = MetricsTracker()
        running_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 记录指标
            running_loss += loss.item()
            tracker.update(loss, outputs, masks)
            
            # 更新进度条
            metrics = tracker.get_avg_metrics()
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'iou': f"{metrics['iou']:.4f}"
            })
        
        avg_metrics = tracker.get_avg_metrics()
        avg_metrics['loss'] = running_loss / len(self.val_loader)
        
        return avg_metrics
    
    def train(self):
        """完整训练流程"""
        epochs = self.config.get('epochs', 50)
        early_stop_patience = self.config.get('early_stop_patience', 10)
        early_stop_counter = 0
        
        print("=" * 60)
        print("开始训练 - 改进U-Net模型")
        print("=" * 60)
        print(f"训练集样本数: {len(self.train_loader.dataset)}")
        print(f"验证集样本数: {len(self.val_loader.dataset)}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {self.config.get('batch_size', 8)}")
        print(f"学习率: {self.config.get('learning_rate', 0.001)}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # TensorBoard记录
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.1f}s")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Dice: {train_metrics['dice']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                early_stop_counter = 0
                
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  -> 保存最佳模型 (Dice: {self.best_dice:.4f})")
            else:
                early_stop_counter += 1
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        total_time = time.time() - start_time
        
        # 训练完成
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证Dice: {self.best_dice:.4f} (Epoch {self.best_epoch})")
        print(f"总训练时间: {total_time / 60:.1f} 分钟")
        print("=" * 60)
        
        # 保存训练历史
        self.save_history()
        
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        """保存训练历史"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_dataloaders_from_config(config: dict) -> dict:
    """根据配置创建数据加载器"""
    data_root = config['data_root']
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    target_size = tuple(config.get('target_size', [512, 512]))
    
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


def main():
    parser = argparse.ArgumentParser(description='训练改进U-Net模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
                        help='数据根目录')
    parser.add_argument('--train_dataset', type=str, default='combined', 
                        help='训练数据集 (first, third, combined)')
    parser.add_argument('--test_dataset', type=str, default='combined',
                        help='测试数据集')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='UNET', choices=['UNET', 'MASK_RCNN', 'YOLO'],
                        help='模型类型')
    parser.add_argument('--base_filters', type=int, default=64,
                        help='基础滤波器数量')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入通道数')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='输出类别数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--output', type=str, 
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints/unet',
                        help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练检查点')
    
    args = parser.parse_args()
    
    # 加载配置
    config = vars(args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    dataloaders = create_dataloaders_from_config(config)
    
    print(f"训练集: {len(dataloaders['train'].dataset)} 样本")
    print(f"验证集: {len(dataloaders['val'].dataset)} 样本")
    print(f"测试集: {len(dataloaders['test'].dataset)} 样本")
    
    # 创建模型
    print("\n创建改进U-Net模型...")
    model = ImprovedUNet(
        in_channels=config['in_channels'],
        out_channels=config['num_classes'],
        base_filters=config['base_filters'],
        deep_supervision=True
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 恢复训练
    if config['resume']:
        print(f"恢复训练: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建损失函数和优化器
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # 开始训练
    trainer.train()
    
    # 最终测试评估
    print("\n最终测试集评估...")
    test_metrics = trainer.validate(config['epochs'])
    print(f"测试集指标:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Dice: {test_metrics['dice']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
