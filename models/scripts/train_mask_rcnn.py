"""
松材线虫病检测系统 - Mask R-CNN训练脚本
"""

import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessing import PineNematodeDataset


def collate_fn(batch):
    """自定义批处理函数"""
    return tuple(zip(*batch))


def get_maskrcnn_model(num_classes=2, pretrained=True):
    """创建Mask R-CNN模型"""
    # 使用ResNet50-FPN骨干网络
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        pretrained=pretrained,
        trainable_layers=3
    )
    
    # 创建Mask R-CNN
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 替换掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 准备图像和目标
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
            
        except Exception as e:
            print(f"Batch {batch_idx} 错误: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    
    # 收集预测结果
    all_predictions = []
    total_detections = 0
    
    pbar = tqdm(data_loader, desc='[Evaluate]')
    
    for images, _ in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for output in outputs:
            all_predictions.append({
                'boxes': output['boxes'].cpu(),
                'labels': output['labels'].cpu(),
                'scores': output['scores'].cpu(),
                'masks': output['masks'].cpu() if 'masks' in output else None
            })
            total_detections += len(output['boxes'])
    
    return {
        'total_detections': total_detections,
        'num_images': len(all_predictions),
        'avg_detections': total_detections / max(len(all_predictions), 1)
    }


def prepare_targets(masks, threshold=0.5):
    """将分割掩码转换为检测目标格式"""
    import cv2
    
    batch_size = masks.shape[0]
    targets = []
    
    for i in range(batch_size):
        mask = masks[i].numpy()
        
        # 查找连通区域
        mask_binary = (mask > threshold).astype('uint8')
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        labels = []
        masks_list = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 5 and h > 5:  # 过滤太小的检测
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # 线虫类别
                
                # 创建实例掩码
                instance_mask = np.zeros_like(mask_binary, dtype=np.uint8)
                cv2.drawContours(instance_mask, [contour], -1, 1, -1)
                masks_list.append(instance_mask)
        
        if len(boxes) == 0:
            # 添加一个虚拟目标以避免空目标错误
            boxes.append([0, 0, 1, 1])
            labels.append(0)  # 背景类
            masks_list.append(np.zeros_like(mask_binary, dtype=np.uint8))
        
        targets.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.tensor(masks_list, dtype=torch.uint8)
        })
    
    return targets


def main():
    parser = argparse.ArgumentParser(description='训练Mask R-CNN模型')
    
    parser.add_argument('--data_root', type=str, 
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/data',
                        help='数据根目录')
    parser.add_argument('--output', type=str,
                        default='D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/checkpoints/mask_rcnn',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_dataset = PineNematodeDataset(args.data_root, split='train')
    val_dataset = PineNematodeDataset(args.data_root, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n创建Mask R-CNN模型...")
    model = get_maskrcnn_model(num_classes=args.num_classes)
    model.to(device)
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 恢复训练
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"恢复训练从 Epoch {start_epoch}")
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n开始训练...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # 学习率调整
        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存检查点
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, output_path / 'best_model.pth')
            print(f"  -> 保存最佳模型")
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, output_path / f'checkpoint_epoch_{epoch}.pth')
    
    print("\n训练完成!")
    
    # 最终评估
    print("\n最终评估...")
    eval_results = evaluate(model, val_loader, device)
    print(f"评估结果: {eval_results}")


if __name__ == '__main__':
    main()
