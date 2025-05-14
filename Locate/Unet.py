import torch
import torch.nn as nn
import torch.nn.functional as F
"""
此代码为改进后Unet网络结构代码
"""

class SEBlock(nn.Module):
    """通道注意力机制(Squeeze-and-Excitation)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class BasicConvBlock(nn.Module):
    """改进的基础卷积块(支持步幅下采样和Dropout)"""
    def __init__(self, in_ch, out_ch, stride=1, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class OptimizedUNet(nn.Module):
    """优化后的轻量级车牌定位网络"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # 编码器（使用步幅卷积下采样）
        self.inc = BasicConvBlock(in_ch, 32, stride=1)
        self.down1 = BasicConvBlock(32, 64, stride=2)
        self.down2 = BasicConvBlock(64, 128, stride=2)
        self.down3 = BasicConvBlock(128, 256, stride=2)

        # 解码器（转置卷积上采样）
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.se1 = SEBlock(128)
        self.conv4 = BasicConvBlock(256, 128, dropout=True)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.se2 = SEBlock(64)
        self.conv5 = BasicConvBlock(128, 64, dropout=True)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.se3 = SEBlock(32)
        self.conv6 = BasicConvBlock(64, 32, dropout=True)


        self.seg_head = nn.Conv2d(32, out_ch, 1)  # 移除Sigmoid
    def forward(self, x):
        # 编码过程
        x1 = self.inc(x)    # [B,32,640,640]
        x2 = self.down1(x1) # [B,64,320,320]
        x3 = self.down2(x2) # [B,128,160,160]
        x4 = self.down3(x3) # [B,256,80,80]

        # 解码过程
        d1 = self.up1(x4)
        x3 = self.se1(x3)
        d1 = torch.cat([d1, x3], dim=1)  # 通道维度拼接
        d1 = self.conv4(d1)  # [B,128,160,160]

        d2 = self.up2(d1)
        x2 = self.se2(x2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv5(d2)  # [B,64,320,320]

        d3 = self.up3(d2)
        x1 = self.se3(x1)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv6(d3)  # [B,32,640,640]

        return self.seg_head(d3)

if __name__ == '__main__':
    model = OptimizedUNet()
    x = torch.randn(2, 3, 640, 640)
    print("输入尺寸:", x.shape)            # [2,3,640,640]
    print("输出尺寸:", model(x).shape)     # [2,1,640,640]
    print("参数量:", sum(p.numel() for p in model.parameters())/1e6, "M")  # 约2.0M