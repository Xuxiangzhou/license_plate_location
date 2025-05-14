import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_size=(512, 512),
                 augment=True, is_train=True):
        """
        Args:
            augment: 是否启用数据增强 (从config.py传入)
            is_train: 是否为训练模式 (验证/测试模式禁用增强)
        """
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.target_size = target_size
        self.augment = augment & is_train  # 双重控制逻辑

        # 数据一致性校验
        assert len(self.img_paths) == len(self.mask_paths), "图像与掩码数量不匹配"
        assert all([os.path.basename(img).split('.')[0] == os.path.basename(mask).split('.')[0]
                    for img, mask in zip(self.img_paths, self.mask_paths)]), "文件名不匹配"
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取原图
        img = cv2.imread(self.img_paths[idx])
        if img is None:  # 新增空值检查
            raise ValueError(f"图像加载失败: {self.img_paths[idx]}")

        # 读取掩码
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:  # 新增掩码检查
            raise ValueError(f"掩码加载失败: {self.mask_paths[idx]}")

        # 读取掩码
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 统一调整尺寸
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_LINEAR)#双线形插值， 2x2 像素邻域的双线性插值方法计算新像素值
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_NEAREST)#最邻近插值，使用最近的像素值进行插值，没有任何权重或计算
        # 数据增强流程
        if self.augment:
            img, mask = self._augment_data(img, mask)

        # 归一化处理
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # 二值化处理

        # 转为PyTorch张量
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> CHW

        return img, mask

    def _augment_data(self, img, mask):

        h, w = img.shape[:2]

        # 随机水平翻转（网页2示例）
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 随机垂直翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # 随机旋转（-15°~15°）（网页1参数）
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # 随机仿射变换
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
            M[0, 1] += shear / 100
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # 颜色增强
        if random.random() > 0.5:
            img = self._color_jitter(img)
        # cv2.waitKey(1)  # 释放OpenCV缓存
        return img, mask

    def _color_jitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2):

        # 亮度调整
        if random.random() < 0.5:
            delta = random.uniform(-brightness, brightness)
            img = cv2.add(img, delta * 255)

        # 对比度调整
        if random.random() < 0.5:
            alpha = random.uniform(1 - contrast, 1 + contrast)
            img = cv2.multiply(img, alpha)

        # 饱和度调整
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(1 - saturation, 1 + saturation)
            hsv = np.clip(hsv, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img