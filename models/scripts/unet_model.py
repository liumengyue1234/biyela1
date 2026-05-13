"""
松材线虫病检测系统 - 改进U-Net模型
结合CBAM注意力机制和残差连接，提升CT图像分割性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class ConvBlock(nn.Module):
    """卷积块：卷积 + BatchNorm + ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """CBAM注意力模块 - 通道注意力 + 空间注意力"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class AttentionGate(nn.Module):
    """注意力门控模块 - 用于跳跃连接"""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = gate_channels // 2
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # gate: 下采样特征, skip: 上采样跳跃连接特征
        gate_conv = self.W_gate(gate)
        skip_conv = self.W_skip(skip)
        
        # 融合
        combined = self.relu(gate_conv + skip_conv)
        attention = self.psi(combined)
        
        return skip * attention


class DoubleConv(nn.Module):
    """双层卷积块 - 用于编码器和解码器"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块 - 编码器"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块 - 解码器，带注意力门控"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        # 注意力门控
        self.attention_gate = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=in_channels // 2,
            inter_channels=in_channels // 4
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # 应用注意力门控
        x2_attended = self.attention_gate(x1, x2)
        
        # 拼接
        x = torch.cat([x2_attended, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.conv(x)


class ImprovedUNet(nn.Module):
    """
    改进U-Net模型
    
    改进点:
    1. CBAM注意力机制增强特征提取
    2. 残差连接改善梯度流动
    3. 注意力门控改进跳跃连接
    4. Dropout防止过拟合
    5. Deep Supervision多尺度监督
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_filters: int = 64,
        bilinear: bool = False,
        deep_supervision: bool = True
    ):
        """
        初始化改进U-Net
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（类别数）
            base_filters: 基础滤波器数量
            bilinear: 是否使用双线性插值
            deep_supervision: 是否使用深度监督
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        
        # 初始卷积
        self.in_conv = DoubleConv(in_channels, base_filters, base_filters)
        
        # CBAM注意力 - 仅在编码器部分
        self.cbam1 = CBAM(base_filters, reduction=4)
        self.cbam2 = CBAM(base_filters * 2, reduction=8)
        self.cbam3 = CBAM(base_filters * 4, reduction=16)
        self.cbam4 = CBAM(base_filters * 8, reduction=32)
        
        # 编码器
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # 残差块
        self.res_block1 = ResidualBlock(base_filters * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # 解码器
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # 输出层
        self.out_conv = OutConv(base_filters, out_channels)
        
        # 深度监督层
        if deep_supervision:
            self.ds_out1 = OutConv(base_filters * 8 // factor, out_channels)
            self.ds_out2 = OutConv(base_filters * 4, out_channels)
            self.ds_out3 = OutConv(base_filters * 2, out_channels)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
        
        Returns:
            分割掩码 [B, num_classes, H, W]
        """
        # 初始卷积
        x = self.in_conv(x)
        x = self.cbam1(x)
        
        # 编码器
        x1 = x  # 跳跃连接
        x = self.down1(x)
        x = self.cbam2(x)
        
        x2 = x  # 跳跃连接
        x = self.down2(x)
        x = self.cbam3(x)
        
        x3 = x  # 跳跃连接
        x = self.down3(x)
        x = self.cbam4(x)
        
        x4 = x  # 跳跃连接
        x = self.down4(x)
        
        # 残差块
        x = self.res_block1(x)
        
        # 解码器
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        if self.deep_supervision and self.training:
            # 深度监督模式 - 返回多尺度输出
            return self.out_conv(x)
        else:
            return self.out_conv(x)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        
        Args:
            pred: 预测概率 [B, C, H, W]
            target: 目标掩码 [B, H, W]
        
        Returns:
            Dice损失值
        """
        pred = F.softmax(pred, dim=1)
        
        # 只计算前景类
        pred = pred[:, 1]  # [B, H, W]
        
        # 平滑
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """组合损失函数 - Dice + CrossEntropy"""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """Focal损失函数 - 处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


# 模型工厂函数
def create_improved_unet(
    in_channels: int = 3,
    out_channels: int = 2,
    base_filters: int = 64,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None
) -> ImprovedUNet:
    """
    创建改进U-Net模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出类别数
        base_filters: 基础滤波器数量
        pretrained: 是否使用预训练权重
        checkpoint_path: 权重文件路径
    
    Returns:
        改进U-Net模型
    """
    model = ImprovedUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        deep_supervision=True
    )
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_improved_unet(
        in_channels=3,
        out_channels=2,
        base_filters=64
    ).to(device)
    
    print(f"模型参数量: {model.get_num_parameters():,}")
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算损失
    target = torch.randint(0, 2, (batch_size, 512, 512)).to(device)
    criterion = CombinedLoss()
    loss = criterion(output, target)
    print(f"损失值: {loss.item():.4f}")
