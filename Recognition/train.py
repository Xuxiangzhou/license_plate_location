import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from dataset import get_dataloader
from CRNN import CRNN
# 车牌字符识别网络CRNN模型训练代码
# -------------------- 超参数配置 --------------------
config = {
    "train_img_dir": "../data/recognization/train",#训练数据集
    "val_img_dir": "../data/recognization/val",#验证数据集，此验证集用于评估训练中每次迭代后模型性能，在据此进行调优，不是评估最终模型性能的测试集
    "train_txt": "../data/recognization/split/train.txt",#训练数据集标签
    "val_txt": "../data/recognization/split/val.txt",#验证数据集标签
    "batch_size": 128,#批量大小
    "lr": 0.001,#初始学习率
    "epochs": 300,#迭代次数
    "device": "cuda" if torch.cuda.is_available() else "cpu",#推理设备选择
    "save_dir": "./checkpoints",#模型保存路径
    "img_height": 32,#图片尺寸
    "img_width": 160,
    "early_stopping_patience": 10  # 早停机制的耐心值
}

# -------------------- 字符到索引的映射 --------------------
CHARS = "京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵青藏川宁琼粤0123456789ABCDEFGHJKLMNPQRSTUVWXYZ挂学警港澳使领"
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
char_to_idx["<blank>"] = 0  # CTC空白符

# -------------------- Focal CTC Loss 实现 --------------------
class FocalCTCLoss(nn.Module):
    def __init__(self, gamma=2):
        """
        Focal CTC Loss
        :param gamma: Focusing parameter, default is 2.
        """
        super(FocalCTCLoss, self).__init__()
        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='none')  # 使用 reduction='none' 得到每个样本的损失

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Forward pass for Focal CTC Loss
        :param log_probs: Log probabilities from the model (log_softmax applied), shape [T, N, C]
        :param targets: Ground truth labels, shape [N, S]
        :param input_lengths: Lengths of inputs (T), shape [N]
        :param target_lengths: Lengths of targets (S), shape [N]
        :return: Focal CTC Loss
        """
        # Standard CTC Loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)  # Shape: [N]

        # Apply Focal Loss weighting
        focal_weight = (1 - torch.exp(-ctc_loss)) ** self.gamma  # Focal weight: (1 - p)^gamma
        focal_ctc_loss = focal_weight * ctc_loss  # Weighted loss
        return focal_ctc_loss.mean()  # Mean reduction


# -------------------- 训练函数 --------------------
def train():
    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)

    # 数据加载器
    train_loader = get_dataloader(config["train_txt"], config["train_img_dir"], char_to_idx,
                                   batch_size=config["batch_size"], phase="train")
    val_loader = get_dataloader(config["val_txt"], config["val_img_dir"], char_to_idx,
                                 batch_size=config["batch_size"], phase="val")

    # 初始化模型
    model = CRNN(num_chars=len(char_to_idx)).to(config["device"])
    criterion = FocalCTCLoss(gamma=2)  # 使用 Focal CTC Loss
    optimizer = Adam(model.parameters(), lr=config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])  # 余弦退火学习率

    # 早停机制变量
    best_val_loss = float("inf")
    early_stopping_counter = 0

    # 损失记录
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for imgs, labels, label_lengths in train_loader:
            imgs, labels, label_lengths = imgs.to(config["device"]), labels.to(config["device"]), label_lengths.to(
                config["device"]
            )

            # 前向传播
            logits = model(imgs)  # [T, B, C]
            log_probs = logits.log_softmax(2)  # CTC需要对logits进行log_softmax
            input_lengths = torch.full((logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(
                config["device"]
            )

            # 计算 Focal CTC 损失
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # 记录训练损失

        # 更新学习率
        scheduler.step()

        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion, config["device"])
        val_losses.append(val_loss)  # 记录验证损失
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_model.pth"))
            print(f"Saved best model with Val Loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"No improvement. Early stopping counter: {early_stopping_counter}/{config['early_stopping_patience']}")

        # 检查早停条件
        if early_stopping_counter >= config["early_stopping_patience"]:
            print("Early stopping triggered. Training stopped.")
            break

    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, config["save_dir"])


# -------------------- 验证函数 --------------------
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels, label_lengths in val_loader:
            imgs, labels, label_lengths = imgs.to(device), labels.to(device), label_lengths.to(device)

            logits = model(imgs)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full((logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# -------------------- 绘制损失曲线函数 --------------------
def plot_loss_curve(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="val", color="orange", marker="o")
    plt.title("loss-curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    train()