"""
松材线虫病检测系统 - 检测脚本
支持U-Net、Mask R-CNN、YOLO三种模型的检测
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# 添加脚本路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unet_model import create_improved_unet


class NematodeDetector:
    """松材线虫检测器"""
    
    def __init__(
        self,
        model_type: str = 'UNET',
        model_path: str = None,
        device: str = None,
        conf_threshold: float = 0.5
    ):
        """
        初始化检测器
        
        Args:
            model_type: 模型类型 UNET, MASK_RCNN, YOLO
            model_path: 模型权重路径
            device: 设备 'cuda' 或 'cpu'
            conf_threshold: 置信度阈值
        """
        self.model_type = model_type.upper()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self.model = self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.model_type == 'UNET':
            model = create_improved_unet(
                in_channels=3,
                out_channels=2,
                base_filters=64
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"加载模型权重: {self.model_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def detect(self, image_path: str) -> Dict:
        """
        检测图像中的松材线虫
        
        Args:
            image_path: 图像路径
        
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_image = image.copy()
        original_shape = image.shape[:2]
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        input_size = (512, 512)
        image_resized = cv2.resize(image, input_size)
        
        # 归一化并转换为张量
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 推理
        output = self.model(image_tensor)
        
        # 后处理
        if self.model_type == 'UNET':
            result = self._process_unet_output(output, original_shape)
        else:
            result = self._process_unet_output(output, original_shape)
        
        result['processing_time'] = time.time() - start_time
        result['image_path'] = image_path
        
        return result
    
    def _process_unet_output(
        self, 
        output: torch.Tensor, 
        original_shape: Tuple[int, int]
    ) -> Dict:
        """处理U-Net输出"""
        # 获取预测
        pred = output.squeeze(0)  # [C, H, W]
        pred_mask = pred.argmax(dim=0)  # [H, W]
        pred_probs = F.softmax(pred, dim=1)[1]  # 前景概率 [H, W]
        
        # 调整回原始尺寸
        pred_mask = cv2.resize(
            pred_mask.cpu().numpy().astype('uint8'),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        pred_probs = cv2.resize(
            pred_probs.cpu().numpy(),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 找到所有连通区域
        contours, _ = cv2.findContours(
            pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        results = []
        nematode_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 50:  # 过滤太小的区域
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算平均置信度
            roi_mask = pred_mask[y:y+h, x:x+w]
            roi_probs = pred_probs[y:y+h, x:x+w]
            avg_conf = float(roi_probs[roi_mask > 0].mean()) if (roi_mask > 0).sum() > 0 else 0
            
            if avg_conf >= self.conf_threshold:
                results.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': round(avg_conf, 4),
                    'area': int(area)
                })
                nematode_count += 1
        
        return {
            'nematode_count': nematode_count,
            'confidence': round(np.mean([r['confidence'] for r in results]) if results else 0, 4),
            'results': results,
            'pred_mask': pred_mask
        }
    
    def visualize(
        self,
        image_path: str,
        result: Dict,
        output_path: str = None
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image_path: 原图路径
            result: 检测结果
            output_path: 输出路径
        
        Returns:
            可视化图像
        """
        image = cv2.imread(image_path)
        
        # 绘制预测掩码（半透明）
        if 'pred_mask' in result:
            mask = result['pred_mask']
            
            # 创建彩色掩码
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色表示检测到的区域
            
            # 叠加
            image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        # 绘制边界框
        for i, det in enumerate(result['results']):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # 绘制边界框
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"#{i+1} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color, -1
            )
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        # 添加统计信息
        info_text = f"Nematodes: {result['nematode_count']} | Confidence: {result['confidence']:.2f}"
        cv2.putText(
            image, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image


def main():
    parser = argparse.ArgumentParser(description='松材线虫检测')
    
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='UNET', 
                       choices=['UNET', 'MASK_RCNN', 'YOLO'], help='模型类型')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--output', type=str, default='result.png', help='输出图像路径')
    parser.add_argument('--save_mask', action='store_true', help='保存预测掩码')
    
    args = parser.parse_args()
    
    # 创建检测器
    print(f"初始化{args.model}检测器...")
    detector = NematodeDetector(
        model_type=args.model,
        model_path=args.weights,
        conf_threshold=args.threshold
    )
    
    # 执行检测
    print(f"检测图像: {args.image}")
    result = detector.detect(args.image)
    
    # 打印结果
    print("\n" + "=" * 40)
    print("检测结果")
    print("=" * 40)
    print(f"检测到的线虫数量: {result['nematode_count']}")
    print(f"平均置信度: {result['confidence']:.4f}")
    print(f"处理时间: {result['processing_time']:.3f}s")
    
    if result['results']:
        print("\n详细信息:")
        for i, det in enumerate(result['results']):
            print(f"  #{i+1}: bbox={det['bbox']}, conf={det['confidence']:.4f}, area={det['area']}")
    
    # 保存可视化结果
    print(f"\n保存可视化结果: {args.output}")
    detector.visualize(args.image, result, args.output)
    
    # 保存检测结果JSON
    json_path = args.output.replace('.png', '.json')
    result_json = {
        'image_path': result['image_path'],
        'model_type': args.model,
        'nematode_count': result['nematode_count'],
        'confidence': result['confidence'],
        'processing_time': result['processing_time'],
        'results': result['results']
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    print(f"保存检测结果JSON: {json_path}")
    
    # 保存掩码
    if args.save_mask and 'pred_mask' in result:
        mask_path = args.output.replace('.png', '_mask.png')
        cv2.imwrite(mask_path, result['pred_mask'] * 255)
        print(f"保存预测掩码: {mask_path}")


if __name__ == '__main__':
    main()
