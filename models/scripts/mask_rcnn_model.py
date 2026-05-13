"""
松材线虫病检测系统 - Mask R-CNN模型
用于实例分割任务
"""

import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


def get_maskrcnn_model(
    num_classes: int = 2,
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3
):
    """
    创建Mask R-CNN模型
    
    Args:
        num_classes: 类别数（包括背景）
        pretrained: 是否使用预训练权重
        pretrained_backbone: 骨干网络是否预训练
        trainable_backbone_layers: 可训练骨干网络层数
    
    Returns:
        Mask R-CNN模型
    """
    # 使用ResNet50-FPN作为骨干网络
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers
    )
    
    # 创建Mask R-CNN模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=False,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
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


class MaskRCNNTrainer:
    """Mask R-CNN训练器"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # 优化器参数
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=config.get('learning_rate', 0.001),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0005)
        )
        
        # 学习率调度
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 3),
            gamma=config.get('lr_gamma', 0.1)
        )
    
    def train_one_epoch(self, data_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        for images, targets in data_loader:
            # 转换为目标格式
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            self.optimizer.zero_grad()
            losses.backward()
            torch.optim.swa.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {losses.item():.4f}")
        
        self.lr_scheduler.step()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        
        # 简单评估：计算检测到的目标数
        total_detections = 0
        num_images = 0
        
        for images, _ in data_loader:
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)
            
            for output in outputs:
                total_detections += len(output['boxes'])
            
            num_images += len(images)
        
        return {
            'total_detections': total_detections,
            'num_images': num_images,
            'avg_detections': total_detections / max(num_images, 1)
        }


def prepare_target(masks, boxes, labels):
    """
    准备Mask R-CNN的目标格式
    
    Args:
        masks: 掩码列表 [N, H, W]
        boxes: 边界框列表 [N, 4] (x1, y1, x2, y2)
        labels: 标签列表 [N]
    
    Returns:
        目标字典
    """
    targets = []
    
    for i in range(len(labels)):
        target = {
            'boxes': torch.as_tensor(boxes[i], dtype=torch.float32),
            'labels': torch.as_tensor(labels[i], dtype=torch.int64),
            'masks': torch.as_tensor(masks[i], dtype=torch.uint8).unsqueeze(0)
        }
        targets.append(target)
    
    return targets


if __name__ == '__main__':
    # 测试模型创建
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = get_maskrcnn_model(num_classes=2)
    model.to(device)
    
    print("Mask R-CNN模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
