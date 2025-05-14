import torch
import torch.nn as nn
import torch.nn.functional as F
"""
此代码定义了Unet网络训练时的损失函数，采用了 Dice Loss + Focal Loss 的组合实现
"""
class DiceFocalLoss(nn.Module):
    """
    Dice Loss + Focal Loss 的组合实现
    Args:
        alpha: Dice Loss 和 Focal Loss 的权重比例
        gamma: Focal Loss 的聚焦参数，控制难易样本的权重
        smooth: Dice Loss 的平滑项，防止分母为零
    """
    def __init__(self, alpha=0.7, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # 输入维度处理
        pred = pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        # 使用 sigmoid 处理预测值 (仅用于 Dice Loss)
        pred_prob = torch.sigmoid(pred)

        # Dice Loss
        intersection = (pred_prob * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_prob.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice

        # Focal Loss (使用 logits 输入)
        focal_weight = (1 - pred_prob).pow(self.gamma) * target + pred_prob.pow(self.gamma) * (1 - target)
        focal_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal_loss = (focal_loss * focal_weight).mean()

        # 组合损失
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss