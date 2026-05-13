"""
松材线虫病检测系统 - 模型评估与对比脚本
对比U-Net、Mask R-CNN、YOLO三种模型的性能
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessing import PineNematodeDataset
from unet_model import create_improved_unet


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def evaluate_unet(
        self,
        model_path: str,
        data_loader: DataLoader,
        num_classes: int = 2
    ) -> Dict:
        """评估U-Net模型"""
        print("\n评估改进U-Net模型...")
        
        # 加载模型
        model = create_improved_unet(
            in_channels=3,
            out_channels=num_classes,
            base_filters=64
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        # 评估指标
        metrics = {
            'dice_scores': [],
            'ious': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'inference_times': []
        }
        
        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='U-Net评估'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 推理
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                if torch.cuda.is_available():
                    start_time.record()
                    outputs = model(images)
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                else:
                    outputs = model(images)
                    inference_time = 0
                
                # 计算指标
                pred = outputs.argmax(dim=1)
                
                metrics['dice_scores'].append(self._compute_dice(pred, masks))
                metrics['ious'].append(self._compute_iou(pred, masks))
                p, r, f = self._compute_prf(pred, masks)
                metrics['precisions'].append(p)
                metrics['recalls'].append(r)
                metrics['f1_scores'].append(f)
                metrics['inference_times'].append(inference_time)
        
        # 计算平均值
        avg_metrics = {
            'dice': np.mean(metrics['dice_scores']),
            'iou': np.mean(metrics['ious']),
            'precision': np.mean(metrics['precisions']),
            'recall': np.mean(metrics['recalls']),
            'f1': np.mean(metrics['f1_scores']),
            'inference_time_ms': np.mean(metrics['inference_times'])
        }
        
        return avg_metrics
    
    def _compute_dice(self, pred, target):
        """计算Dice系数"""
        pred = pred.float()
        target = target.float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        if union > 0:
            return (2. * intersection / union).item()
        return 0.0
    
    def _compute_iou(self, pred, target, num_classes=2):
        """计算IoU"""
        ious = []
        
        for c in range(1, num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            if union > 0:
                ious.append((intersection / union).item())
        
        return np.mean(ious) if ious else 0.0
    
    def _compute_prf(self, pred, target):
        """计算精确率、召回率、F1"""
        pred_pos = (pred == 1).float()
        target_pos = (target == 1).float()
        
        tp = (pred_pos * target_pos).sum().item()
        fp = (pred_pos * (1 - target_pos)).sum().item()
        fn = ((1 - pred_pos) * target_pos).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def compare_models(
        self,
        data_root: str,
        model_paths: Dict[str, str],
        output_dir: str = None
    ) -> Dict:
        """对比多个模型"""
        print("\n" + "=" * 60)
        print("模型性能对比评估")
        print("=" * 60)
        
        # 创建数据加载器
        test_dataset = PineNematodeDataset(data_root, split='test')
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        
        comparison_results = {}
        
        # 评估每个模型
        for model_type, model_path in model_paths.items():
            print(f"\n评估 {model_type}...")
            
            if model_type.upper() == 'UNET':
                results = self.evaluate_unet(model_path, test_loader)
                comparison_results[model_type] = results
            # 可扩展其他模型
        
        # 生成对比报告
        report = self._generate_comparison_report(comparison_results)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'comparison_results.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            with open(output_path / 'comparison_report.md', 'w') as f:
                f.write(report)
        
        return comparison_results, report
    
    def _generate_comparison_report(self, results: Dict) -> str:
        """生成对比报告"""
        report = []
        report.append("# 松材线虫检测模型性能对比报告\n")
        report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n\n")
        
        report.append("## 性能指标对比\n\n")
        report.append("| 模型 | Dice | IoU | Precision | Recall | F1-Score | 推理时间(ms) |\n")
        report.append("|------|------|-----|-----------|--------|----------|---------------|\n")
        
        for model_name, metrics in results.items():
            report.append(f"| {model_name} | {metrics['dice']:.4f} | {metrics['iou']:.4f} | "
                         f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                         f"{metrics['f1']:.4f} | {metrics['inference_time_ms']:.2f} |\n")
        
        report.append("\n## 详细分析\n\n")
        
        # 找出最佳模型
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['dice'])
            report.append(f"**最佳模型**: {best_model[0]} (Dice: {best_model[1]['dice']:.4f})\n\n")
            
            report.append("### 各模型特点\n\n")
            
            for model_name, metrics in results.items():
                report.append(f"#### {model_name}\n")
                report.append(f"- Dice系数: {metrics['dice']:.4f}\n")
                report.append(f"- IoU: {metrics['iou']:.4f}\n")
                report.append(f"- 精确率: {metrics['precision']:.4f}\n")
                report.append(f"- 召回率: {metrics['recall']:.4f}\n")
                report.append(f"- F1分数: {metrics['f1']:.4f}\n")
                
                if model_name.upper() == 'UNET':
                    report.append("\n**改进U-Net特点**:\n")
                    report.append("- 结合CBAM注意力机制，增强特征提取能力\n")
                    report.append("- 残差连接改善梯度流动\n")
                    report.append("- 注意力门控改进跳跃连接\n")
                
                report.append("\n")
        
        return "".join(report)


def main():
    parser = argparse.ArgumentParser(description='模型评估与对比')
    
    parser.add_argument('--data_root', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
                        help='测试数据根目录')
    parser.add_argument('--output', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/results',
                        help='输出目录')
    parser.add_argument('--unet_weights', type=str, default=None,
                        help='U-Net模型权重路径')
    parser.add_argument('--mask_rcnn_weights', type=str, default=None,
                        help='Mask R-CNN模型权重路径')
    parser.add_argument('--yolo_weights', type=str, default=None,
                        help='YOLO模型权重路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建评估器
    evaluator = ModelEvaluator(device=device)
    
    # 配置模型路径
    model_paths = {}
    
    if args.unet_weights:
        model_paths['UNET'] = args.unet_weights
    else:
        # 默认路径
        default_unet = Path('D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints/unet/best_model.pth')
        if default_unet.exists():
            model_paths['UNET'] = str(default_unet)
    
    if not model_paths:
        print("Warning: 没有找到任何模型权重，将使用随机初始化模型进行评估演示")
        model_paths['UNET'] = None
    
    # 执行评估
    comparison_results, report = evaluator.compare_models(
        data_root=args.data_root,
        model_paths=model_paths,
        output_dir=args.output
    )
    
    # 打印报告
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    print(report)
    
    print(f"\n报告已保存至: {args.output}")


if __name__ == '__main__':
    main()
