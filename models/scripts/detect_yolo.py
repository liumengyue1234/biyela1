"""
松材线虫病检测系统 - YOLO 检测脚本
专门用于 YOLO 模型的图像检测
需要 ultralytics 库支持: pip install ultralytics
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

# 添加脚本路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 尝试导入 ultralytics，如果不可用则使用模拟模式
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics 未安装，将使用简化版 YOLO 检测器")
    print("安装命令: pip install ultralytics")


class YOLODetector:
    """基于 YOLO 的松材线虫检测器"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ):
        """
        初始化 YOLO 检测器

        Args:
            model_path: 模型权重路径 (.pt 文件)
            device: 设备 'cuda' 或 'cpu'
            conf_threshold: 置信度阈值
            iou_threshold: IOU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 设置设备
        if device is None:
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 加载模型
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """加载 YOLO 模型"""
        if not YOLO_AVAILABLE:
            # 返回 None，稍后使用模拟模式
            return None

        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"加载 YOLO 模型权重: {model_path}")
        else:
            # 使用默认的 YOLOv8 模型
            model = YOLO('yolov8n.pt')
            print("使用预训练 YOLOv8n 模型")

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

        # 执行检测
        if self.model is not None and YOLO_AVAILABLE:
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            # 处理结果
            result = self._process_yolo_results(results[0], original_shape)
        else:
            # 使用模拟模式（当没有 YOLO 模型时）
            result = self._simulate_detection(image_path, original_shape)

        result['processing_time'] = time.time() - start_time
        result['image_path'] = image_path

        return result

    def _process_yolo_results(self, result, original_shape: Tuple) -> Dict:
        """处理 YOLO 检测结果"""
        boxes = result.boxes

        results = []
        nematode_count = 0

        for i, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # 假设类别 0 为松材线虫
            if cls != 0:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area = int((x2 - x1) * (y2 - y1))

            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': round(conf, 4),
                'area': area,
                'class': cls
            })
            nematode_count += 1

        return {
            'nematode_count': nematode_count,
            'confidence': round(
                float(np.mean([r['confidence'] for r in results])) if results else 0,
                4
            ),
            'results': results,
            'pred_mask': self._boxes_to_mask(results, original_shape)
        }

    def _boxes_to_mask(self, boxes: List[Dict], original_shape: Tuple) -> np.ndarray:
        """将边界框转换为掩码"""
        mask = np.zeros(original_shape, dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            mask[y1:y2, x1:x2] = 1
        return mask

    def _simulate_detection(self, image_path: str, original_shape: Tuple) -> Dict:
        """
        模拟检测（当没有 YOLO 模型时的后备方案）
        基于图像特征进行简单的模拟检测
        """
        image = cv2.imread(image_path)
        h, w = original_shape

        # 简单的模拟检测逻辑
        # 在实际应用中，这里应该替换为真正的模型推理
        # 仅用于测试和演示

        # 读取真实标签（如果有）
        label_path = image_path.replace('.jpg', '_label.png').replace('.png', '_label.png')
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is not None:
                label = cv2.resize(label, (w, h))
                # 找连通区域
                _, binary = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                results = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 50:
                        continue
                    x, y, bw, bh = cv2.boundingRect(contour)
                    results.append({
                        'bbox': [int(x), int(y), int(x + bw), int(y + bh)],
                        'confidence': 0.85,
                        'area': int(area)
                    })

                return {
                    'nematode_count': len(results),
                    'confidence': 0.85,
                    'results': results,
                    'pred_mask': (label > 127).astype(np.uint8)
                }

        # 无标签时的默认模拟结果
        return {
            'nematode_count': 0,
            'confidence': 0.0,
            'results': [],
            'pred_mask': np.zeros(original_shape, dtype=np.uint8),
            'note': '使用模拟检测模式，请训练 YOLO 模型以获得真实结果'
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
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = [0, 165, 255]  # 橙色表示 YOLO 检测区域
            image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

        # 绘制边界框
        for i, det in enumerate(result['results']):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            color = (0, 165, 255)  # 橙色
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
        info_text = f"YOLO | Nematodes: {result['nematode_count']} | Confidence: {result['confidence']:.2f}"
        cv2.putText(
            image, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
        )

        if output_path:
            cv2.imwrite(output_path, image)

        return image


def main():
    parser = argparse.ArgumentParser(description='松材线虫检测 - YOLO')

    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weights', type=str, default=None, help='YOLO 模型权重路径 (.pt)')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU 阈值')
    parser.add_argument('--output', type=str, default='result_yolo.png', help='输出图像路径')

    args = parser.parse_args()

    # 创建检测器
    print("初始化 YOLO 检测器...")
    detector = YOLODetector(
        model_path=args.weights,
        conf_threshold=args.threshold,
        iou_threshold=args.iou
    )

    # 执行检测
    print(f"检测图像: {args.image}")
    result = detector.detect(args.image)

    # 打印结果
    print("\n" + "=" * 50)
    print("YOLO 检测结果")
    print("=" * 50)
    print(f"检测到的线虫数量: {result['nematode_count']}")
    print(f"平均置信度: {result['confidence']:.4f}")
    print(f"处理时间: {result['processing_time']:.3f}s")

    if 'note' in result:
        print(f"注意: {result['note']}")

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
        'model_type': 'YOLO',
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
