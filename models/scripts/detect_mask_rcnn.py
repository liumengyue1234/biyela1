"""
松材线虫病检测系统 - Mask R-CNN 检测脚本
专门用于 Mask R-CNN 模型的图像检测
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
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 添加脚本路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


class MaskRCNNDetector:
    """基于 Mask R-CNN 的松材线虫检测器"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ):
        """
        初始化 Mask R-CNN 检测器

        Args:
            model_path: 模型权重路径
            device: 设备 'cuda' 或 'cpu'
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """加载 Mask R-CNN 模型"""
        # 加载预训练模型
        model = maskrcnn_resnet50_fpn(pretrained=True)

        # 获取分类器的输入特征数
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # 替换预训练头
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        # 获取掩码预测器的输入特征数
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # 替换掩码预测器
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes=2
        )

        # 加载自定义权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"加载模型权重: {model_path}")

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

        # 转换为 RGB 并转换为张量
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # 准备输入
        image_list = [image_tensor.to(self.device)]
        image_height, image_width = original_shape

        # 推理
        predictions = self.model(image_list)

        # 处理预测结果
        result = self._process_predictions(predictions[0], original_shape)

        result['processing_time'] = time.time() - start_time
        result['image_path'] = image_path

        return result

    def _process_predictions(
        self,
        prediction: Dict,
        original_shape: Tuple[int, int]
    ) -> Dict:
        """处理 Mask R-CNN 预测结果"""
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy()

        results = []
        nematode_count = 0

        for i in range(len(boxes)):
            score = scores[i]

            if score < self.conf_threshold:
                continue

            # 获取边界框
            box = boxes[i]
            x1, y1, x2, y2 = box

            # 缩放到原始图像尺寸
            h, w = original_shape
            scale_x = w / masks.shape[2]
            scale_y = h / masks.shape[1]

            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            # 获取掩码
            mask = masks[i, 0] > 0.5
            mask_resized = cv2.resize(
                mask.astype('uint8'),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            # 计算掩码面积
            area = np.sum(mask_resized)

            if area < 50:  # 过滤太小的区域
                continue

            results.append({
                'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                'confidence': round(float(score), 4),
                'area': int(area),
                'mask': mask_resized
            })
            nematode_count += 1

        return {
            'nematode_count': nematode_count,
            'confidence': round(
                float(np.mean([r['confidence'] for r in results])) if results else 0,
                4
            ),
            'results': [{k: v for k, v in r.items() if k != 'mask'} for r in results],
            'all_masks': [r['mask'] for r in results],
            'pred_mask': self._combine_masks([r['mask'] for r in results], original_shape)
        }

    def _combine_masks(self, masks: List[np.ndarray], original_shape: Tuple) -> np.ndarray:
        """合并所有掩码"""
        combined = np.zeros(original_shape, dtype=np.uint8)
        for mask in masks:
            combined = np.maximum(combined, mask)
        return combined

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
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = [255, 0, 0]  # 蓝色表示 Mask R-CNN 检测区域
            image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

        # 绘制边界框
        for i, det in enumerate(result['results']):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            color = (255, 0, 0)  # 蓝色
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # 添加统计信息
        info_text = f"Mask R-CNN | Nematodes: {result['nematode_count']} | Confidence: {result['confidence']:.2f}"
        cv2.putText(
            image, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        if output_path:
            cv2.imwrite(output_path, image)

        return image


def main():
    parser = argparse.ArgumentParser(description='松材线虫检测 - Mask R-CNN')

    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--nms', type=float, default=0.5, help='NMS 阈值')
    parser.add_argument('--output', type=str, default='result_maskrcnn.png', help='输出图像路径')

    args = parser.parse_args()

    # 创建检测器
    print("初始化 Mask R-CNN 检测器...")
    detector = MaskRCNNDetector(
        model_path=args.weights,
        conf_threshold=args.threshold,
        nms_threshold=args.nms
    )

    # 执行检测
    print(f"检测图像: {args.image}")
    result = detector.detect(args.image)

    # 打印结果
    print("\n" + "=" * 50)
    print("Mask R-CNN 检测结果")
    print("=" * 50)
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
        'model_type': 'MASK_RCNN',
        'nematode_count': result['nematode_count'],
        'confidence': result['confidence'],
        'processing_time': result['processing_time'],
        'results': result['results']
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"保存检测结果JSON: {json_path}")


if __name__ == '__main__':
    main()
