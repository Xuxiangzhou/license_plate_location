import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights

"""
此代码为改进后CRNN网络结构代码
"""
class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_weights


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=512, pretrained=True):
        super().__init__()

        # -------------------- 增强的 CNN 主干 --------------------
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet34(weights=weights)

        # 使用 ResNet 的浅层层级以保留空间信息
        self.cnn = nn.Sequential(*list(resnet.children())[:-4])  # 输出: [B, 128, H/4, W/4]

        # 用深度可分离卷积替换最后的卷积块
        self.depthwise_conv = nn.Sequential(
            DepthwiseSeparableConv(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 添加 CBAM 模块以改进注意力机制
        self.cbam = CBAM(channel=128)

        # -------------------- 尺寸自适应 --------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # 将高度固定为 1

        # -------------------- Transformer 编码器 --------------------
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,  # 与 CNN 输出通道匹配
                nhead=8,  # 注意力头的数量
                dim_feedforward=hidden_size,
                dropout=0.3
            ),
            num_layers=4  # Transformer 层的数量
        )

        # -------------------- 输出层 --------------------
        self.fc = nn.Linear(128, num_chars)

    def forward(self, x):
        # 通过 CNN 提取特征
        x = self.cnn(x)  # [B, 128, H/4, W/4]
        x = self.depthwise_conv(x)  # [B, 128, H/4, W/4]

        # 应用 CBAM 模块
        x = self.cbam(x)

        # 尺寸自适应
        x = self.adaptive_pool(x)  # [B, 128, 1, W/4]
        x = x.squeeze(2)  # [B, 128, W/4]
        x = x.permute(2, 0, 1)  # [W/4, B, 128]

        # 使用 Transformer 编码器进行序列建模
        x = self.transformer(x)  # [W/4, B, 128]

        # 字符分类
        logits = self.fc(x)  # [W/4, B, num_chars]
        return logits


if __name__ == "__main__":
    # -------------------- 测试模型 --------------------
    dummy_input = torch.randn(4, 3, 32, 128)  # 批量大小 = 4，图像尺寸 = 32x128 (H x W)
    model = CRNN(num_chars=68)  # 假设字符类别数量为 68

    print("输入形状:", dummy_input.shape)  # [4, 3, 32, 128]

    # 测试 CNN 输出
    cnn_out = model.cnn(dummy_input)
    print("CNN 输出形状:", cnn_out.shape)  # [4, 128, 8, 32]

    # 测试深度可分离卷积
    depthwise_out = model.depthwise_conv(cnn_out)
    print("深度可分离卷积输出形状:", depthwise_out.shape)  # [4, 128, 8, 32]

    # 测试自适应池化
    pooled = model.adaptive_pool(depthwise_out)
    print("池化输出形状:", pooled.shape)  # [4, 128, 1, 32]

    # 测试完整前向传播
    logits = model(dummy_input)
    print("最终输出形状:", logits.shape)  # [32, 4, 68] (序列长度, 批量大小, 字符类别)