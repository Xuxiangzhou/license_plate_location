# 本代码用于生成Unet网络计算图，仅作演示使用，非核心代码
from torchviz import make_dot

from Locate.Unet import OptimizedUNet
import torch
if __name__ == "__main__":
    model = OptimizedUNet()
    x = torch.randn(2, 3, 640, 640)
    y = model(x)  # 前向传播
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("optimized_unet", format="png")  # 生成 PNG 图片
    