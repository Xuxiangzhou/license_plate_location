import torch
import torch.nn as nn
"""
此代码为传统Unet网络结构代码
"""

class OptimizedUNet(nn.Module):
    """传统U-Net网络"""

    def __init__(self, in_ch=3, out_ch=1):
        super(OptimizedUNet, self).__init__()

        # 编码器部分
        self.enc_conv1 = self.double_conv(in_ch, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)

        # 解码器部分
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)

        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)

        # 最后一层输出
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        """两个卷积层 + ReLU 激活"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码过程
        x1 = self.enc_conv1(x)  # [B,64,H,W]
        x2 = nn.MaxPool2d(2)(x1)  # 下采样 [B,64,H/2,W/2]

        x3 = self.enc_conv2(x2)  # [B,128,H/2,W/2]
        x4 = nn.MaxPool2d(2)(x3)  # 下采样 [B,128,H/4,W/4]

        x5 = self.enc_conv3(x4)  # [B,256,H/4,W/4]
        x6 = nn.MaxPool2d(2)(x5)  # 下采样 [B,256,H/8,W/8]

        x7 = self.enc_conv4(x6)  # [B,512,H/8,W/8]

        # 解码过程
        d3 = self.up_conv3(x7)  # 上采样 [B,256,H/4,W/4]
        d3 = torch.cat([d3, x5], dim=1)  # 跳跃连接 [B,512,H/4,W/4]
        d3 = self.dec_conv3(d3)  # [B,256,H/4,W/4]

        d2 = self.up_conv2(d3)  # 上采样 [B,128,H/2,W/2]
        d2 = torch.cat([d2, x3], dim=1)  # 跳跃连接 [B,256,H/2,W/2]
        d2 = self.dec_conv2(d2)  # [B,128,H/2,W/2]

        d1 = self.up_conv1(d2)  # 上采样 [B,64,H,W]
        d1 = torch.cat([d1, x1], dim=1)  # 跳跃连接 [B,128,H,W]
        d1 = self.dec_conv1(d1)  # [B,64,H,W]

        out = self.out_conv(d1)  # 输出 [B,out_ch,H,W]
        return out


if __name__ == '__main__':
    model = OptimizedUNet()
    x = torch.randn(2, 3, 640, 640)
    print("输入尺寸:", x.shape)  # [2,3,640,640]
    print("输出尺寸:", model(x).shape)  # [2,1,640,640]
    print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")  # 参数量