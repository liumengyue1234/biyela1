"""
松材线虫病检测系统 - YOLO v8/v10/v11 对比检测脚本
"""

import sys
import argparse
from pathlib import Path
import time
import json
import cv2
import numpy as np


class YOLOCompareDetector:
    """YOLO多版本对比检测器"""

    def __init__(self, checkpoint_dir: str = None):
        """
        初始化检测器

        Args:
            checkpoint_dir: 模型权重目录
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')

        # YOLO版本配置
        self.versions = {
            'YOLOv8': {'model_type': 'yolov8s', 'config_key': 'v8'},
            'YOLO11': {'model_type': 'yolo11m', 'config_key': 'v11'},
            'YOLOv10': {'model_type': 'yolov10s', 'config_key': 'v10'},
        }

        self.models = {}
        self._load_models()

    def _load_models(self):
        """加载各版本YOLO模型"""
        try:
            from ultralytics import YOLO
            self.ultralytics_available = True
        except ImportError:
            print("Warning: ultralytics未安装，使用模拟模式")
            self.ultralytics_available = False
            return

        for version_name, config in self.versions.items():
            model_type = config['model_type']

            # 尝试加载模型权重
            weights_path = self.checkpoint_dir / model_type / 'weights' / 'best.pt'
            if not weights_path.exists():
                weights_path = self.checkpoint_dir / 'yolo' / 'weights' / 'best.pt'

            if weights_path.exists():
                try:
                    self.models[version_name] = YOLO(str(weights_path))
                    print(f"✓ {version_name} 模型已加载")
                except Exception as e:
                    print(f"✗ {version_name} 加载失败: {e}")
            else:
                print(f"⚠ {version_name} 权重文件不存在，将使用预训练模型")
                try:
                    self.models[version_name] = YOLO(f'{model_type}.pt')
                    print(f"✓ {version_name} 预训练模型已加载")
                except Exception as e:
                    print(f"✗ {version_name} 加载失败: {e}")

    def detect_single(self, image_path: str, version: str = 'YOLOv8', conf_threshold: float = 0.25):
        """
        使用指定版本检测单张图像

        Args:
            image_path: 图像路径
            version: YOLO版本
            conf_threshold: 置信度阈值

        Returns:
            检测结果
        """
        if version not in self.models:
            return {'error': f'{version} 模型未加载', 'boxes': [], 'time': 0}

        if not Path(image_path).exists():
            return {'error': f'图像不存在: {image_path}', 'boxes': [], 'time': 0}

        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': '图像读取失败', 'boxes': [], 'time': 0}

        start_time = time.time()
        results = self.models[version](img, conf=conf_threshold, verbose=False)
        detect_time = time.time() - start_time

        # 解析结果
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                boxes.append({
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3]),
                    'conf': float(box.conf[0]),
                    'cls': int(box.cls[0])
                })

        return {
            'version': version,
            'image': str(image_path),
            'boxes': boxes,
            'count': len(boxes),
            'avg_conf': np.mean([b['conf'] for b in boxes]) if boxes else 0,
            'time': detect_time
        }

    def detect_compare(self, image_path: str, conf_threshold: float = 0.25):
        """
        使用所有版本检测并对比

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值

        Returns:
            对比结果
        """
        results = {}

        for version_name in self.versions.keys():
            result = self.detect_single(image_path, version_name, conf_threshold)
            results[version_name] = result

        return results

    def visualize_comparison(self, image_path: str, results: dict, output_path: str = None):
        """
        可视化对比结果

        Args:
            image_path: 图像路径
            results: 检测结果
            output_path: 输出路径
        """
        import matplotlib.pyplot as plt

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        n_versions = len(results)
        fig, axes = plt.subplots(1, n_versions + 1, figsize=(4 * (n_versions + 1), 4))

        # 原图
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # 各版本结果
        colors = {'YOLOv8': 'red', 'YOLO11': 'green', 'YOLOv10': 'blue'}

        for idx, (version, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            display_img = img.copy()

            if 'boxes' in result and result['boxes']:
                for box in result['boxes']:
                    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    cv2.rectangle(display_img, (x1, y1), (x2, y2),
                                  self._hex_to_rgb(colors.get(version, 'red')), 2)
                    label = f"{box['conf']:.2f}"
                    cv2.putText(display_img, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               self._hex_to_rgb(colors.get(version, 'red')), 2)

            ax.imshow(display_img)
            ax.set_title(f"{version}\nCount: {result.get('count', 0)}\nTime: {result.get('time', 0):.3f}s",
                        fontsize=11)
            ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存: {output_path}")
        else:
            plt.show()

        plt.close()

    def _hex_to_rgb(self, color):
        """将颜色名称转换为RGB"""
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
        }
        return color_map.get(color, (255, 0, 0))


def main():
    parser = argparse.ArgumentParser(description='YOLO多版本对比检测')

    parser.add_argument('--image', '-i', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('--checkpoint_dir', '-c', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints',
                        help='模型权重目录')
    parser.add_argument('--output', '-o', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/outputs',
                        help='输出目录')
    parser.add_argument('--versions', type=str, default='all',
                        help='要对比的版本 (all/v8/v11/v10)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化对比图')

    args = parser.parse_args()

    # 创建检测器
    detector = YOLOCompareDetector(checkpoint_dir=args.checkpoint_dir)

    # 对比检测
    print(f"\n{'='*60}")
    print(f"图像: {args.image}")
    print(f"置信度阈值: {args.conf}")
    print(f"{'='*60}\n")

    results = detector.detect_compare(args.image, conf_threshold=args.conf)

    # 输出结果
    print("检测结果对比:")
    print("-" * 60)

    for version, result in results.items():
        if 'error' in result:
            print(f"{version}: 错误 - {result['error']}")
        else:
            print(f"{version}:")
            print(f"  - 检测数量: {result['count']}")
            print(f"  - 平均置信度: {result['avg_conf']:.4f}")
            print(f"  - 检测时间: {result['time']:.4f}s")
            print()

    # 可视化
    if args.visualize:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        output_path = Path(args.output) / 'yolo_comparison.png'
        detector.visualize_comparison(args.image, results, str(output_path))

    # 保存JSON结果
    json_path = Path(args.output) / 'yolo_comparison.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {json_path}")


if __name__ == '__main__':
    main()
