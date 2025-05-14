import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # 用于显示进度条
import matplotlib.pyplot as plt
from matplotlib import rcParams  # 设置中文字体

from config import Config
from dataset import CustomDataset  # 从您的数据集文件中导入
# from Locate.Unet import OptimizedUNet  # 从您的模型文件中导入
from Locate.basic_Unet import OptimizedUNet
from loss import DiceFocalLoss  # 从您定义的损失函数中导入

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
"""
训练代码
"""

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = Config.num_workers
    epochs = Config.epochs
    batch_size = Config.batch_size
    lr = Config.lr
    weight_decay = 1e-5
    train_img_dir = Config.train_image_dir
    train_mask_dir = Config.train_mask_dir
    val_img_dir = Config.val_image_dir
    val_mask_dir =Config.val_mask_dir
    save_dir = "./checkpoints"
    model_save_dir = "./checkpoints/models"
    target_size = Config.img_size


# 训练和验证代码
def train_validate():
    config = Config()

    # 准备训练数据集和数据加载器
    train_dataset = CustomDataset(
        img_dir=config.train_img_dir,
        mask_dir=config.train_mask_dir,
        target_size=config.target_size,
        augment=True,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 准备验证数据集和数据加载器
    val_dataset = CustomDataset(
        img_dir=config.val_img_dir,
        mask_dir=config.val_mask_dir,
        target_size=config.target_size,
        augment=False,
        is_train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 初始化模型
    model = OptimizedUNet().to(config.device)
    if torch.cuda.device_count() > 1:  # 多GPU训练支持
        model = nn.DataParallel(model)

    # 定义优化器、损失函数和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = DiceFocalLoss(alpha=0.7).to(config.device)  # 自定义损失函数
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 混合精度训练支持
    scaler = torch.amp.GradScaler('cuda')

    # 训练状态记录
    best_val_loss = float('inf')
    best_epoch = -1

    # 损失记录用于绘制曲线
    train_losses = []
    val_losses = []

    # 确保模型保存目录存在
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)

    for epoch in range(config.epochs):
        # 创建一个 epoch 的进度条
        epoch_progress = tqdm(total=len(train_loader) + len(val_loader),
                               desc=f"Epoch {epoch + 1}/{config.epochs}",
                               unit="batch")

        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(config.device), masks.to(config.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # 混合精度训练
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

            # 更新进度条
            epoch_progress.update(1)
            epoch_progress.set_postfix({"训练损失": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.device), masks.to(config.device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # 更新进度条
                epoch_progress.update(1)
                epoch_progress.set_postfix({"验证损失": f"{loss.item():.4f}"})

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 动态学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(config.save_dir, f"best_model_epoch_{best_epoch}.pth")
            torch.save(model.state_dict(), best_model_path)

        # 每 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join(config.model_save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)

        # 更新进度条的最终状态
        epoch_progress.set_postfix({
            "训练损失": f"{train_loss:.4f}",
            "验证损失": f"{val_loss:.4f}",
            "学习率": f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        epoch_progress.close()  # 关闭进度条

    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, config.save_dir)

    print(f"训练完成！最佳模型保存为: {best_model_path} (Epoch {best_epoch})")


# 绘制损失曲线
def plot_loss_curve(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="val_loss", color="orange", marker="o")
    plt.title("loss_curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    train_validate()