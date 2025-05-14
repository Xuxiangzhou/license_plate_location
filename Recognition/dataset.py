import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
"""
数据加载代码
"""

class LicensePlateDataset(Dataset):
    def __init__(self, txt_path, img_dir, char_to_idx, img_height=32, img_width=160, phase="train"):
        """
        初始化车牌数据集
        :param txt_path: 包含图片路径和标签的文本文件路径
        :param img_dir: 图片所在目录
        :param char_to_idx: 字符到索引的映射
        :param img_height: 输入图像的高度（默认32）
        :param img_width: 输入图像的宽度（默认160）
        :param phase: 数据集用途（"train" 或 "val"）
        """
        self.img_dir = img_dir
        self.img_height = img_height
        self.img_width = img_width
        self.phase = phase

        # 读取数据列表
        self.data = self._read_txt(txt_path)

        # 字符到索引的映射
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.num_classes = len(self.char_to_idx)

        # 数据预处理
        if self.phase == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_height + 10, img_width + 10)),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomRotation(10),  # 减小随机旋转角度
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _read_txt(self, txt_path):
        """
        从文本文件读取数据
        :param txt_path: 文本文件路径
        :return: [(图片路径, 标签), ...]
        """
        data = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                image_path, plate_number = line.strip().split(",")
                data.append((image_path, plate_number))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据项
        :param idx: 数据索引
        :return: (图像张量，标签索引列表，标签长度)
        """
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # 确保 image 是 RGB 格式
        img = self.transform(img)
        label_indices = [self.char_to_idx[char] for char in label]
        return img, torch.tensor(label_indices), len(label_indices)


def collate_fn(batch):
    """
    处理变长序列，填充标签
    :param batch: 一个批次的数据
    :return: (图像张量，填充后的标签索引张量，标签长度张量)
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_lengths = [item[2] for item in batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return torch.stack(images), labels_padded, torch.tensor(label_lengths)


def get_dataloader(txt_path, img_dir, char_to_idx, batch_size=32, phase="train"):
    """
    获取数据加载器
    :param txt_path: 包含图片路径和标签的文本文件路径
    :param img_dir: 图片所在目录
    :param char_to_idx: 字符到索引的映射
    :param batch_size: 批次大小
    :param phase: 数据集用途（"train" 或 "val"）
    :return: 数据加载器
    """
    dataset = LicensePlateDataset(txt_path, img_dir, char_to_idx, phase=phase)
    shuffle = True if phase == "train" else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)