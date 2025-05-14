import os
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
from config import Config
from dataset import CustomDataset
from Locate.Unet import OptimizedUNet


class ModelValidator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path).to(self.device)
        self.test_loader = self._init_dataloader()
        os.makedirs(Config.val_results_dir, exist_ok=True)  # 结果保存路径

        # 归一化参数（与训练配置一致）
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def _load_model(self, path):
        """安全加载多输出模型"""
        model = OptimizedUNet(in_ch=3, out_ch=1)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # 兼容单 GPU 和多 GPU 的权重
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v  # 去除多 GPU 前缀

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model

    def _init_dataloader(self):
        """初始化验证数据加载器"""
        return DataLoader(
            CustomDataset(
                img_dir=Config.test_image_dir,
                mask_dir=Config.test_mask_dir,
                target_size=Config.img_size,
                augment=False,
                is_train=False
            ),
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True
        )

    def _denormalize(self, tensor):
        """反归一化处理"""
        img = tensor.cpu().numpy()
        img = img * self.std + self.mean
        img = np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        return img  # 返回 RGB 格式图像，不再转换为 BGR

    def _postprocess_mask(self, pred_mask):
        binary = (pred_mask > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        return cv2.morphologyEx(
            cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
            cv2.MORPH_OPEN, kernel
        )

    def _find_contours(self, processed_mask):
        """轮廓检测"""
        contours, _ = cv2.findContours(
            processed_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    def evaluate(self):
        """执行完整验证流程"""
        all_metrics = {
            "iou": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "ap": []  # 新增AP计算
        }
        total_time = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(tqdm(self.test_loader, desc="Validating", ncols=80)):
                imgs = imgs.to(self.device)
                masks = masks.numpy()  # Ground truth masks

                # 前向推理并记录时间
                start_time = time.time()
                seg_output = self.model(imgs)
                end_time = time.time()

                preds = torch.sigmoid(seg_output).cpu().numpy()

                # 记录推理时间
                total_time += (end_time - start_time)
                num_samples += imgs.size(0)

                # 计算指标
                for i in range(preds.shape[0]):  # 遍历 batch 内的每个样本
                    y_pred = preds[i].flatten()
                    y_true = masks[i].flatten()
                    binary_pred = (y_pred > 0.5)  # 默认阈值为 0.5

                    if binary_pred.sum() == 0:  # 检查是否存在无正样本预测
                        print(f"Warning: No predicted positive samples in batch {batch_idx}, sample {i}.")
                        print(f"Ground truth positive samples: {y_true.sum()}")

                    all_metrics["iou"].append(self._calculate_iou(y_true, binary_pred))
                    all_metrics["accuracy"].append(accuracy_score(y_true, binary_pred))
                    all_metrics["precision"].append(precision_score(y_true, binary_pred, zero_division=1))
                    all_metrics["recall"].append(recall_score(y_true, binary_pred, zero_division=1))
                    all_metrics["f1"].append(f1_score(y_true, binary_pred, zero_division=1))
                    all_metrics["ap"].append(average_precision_score(y_true, y_pred))  # 添加AP计算

                # 可视化保存
                if batch_idx % 10 == 0:
                    self._visualize_sample(imgs[0], masks[0], preds[0], batch_idx)

                # 内存优化：及时释放变量
                del imgs, masks, seg_output, preds
                torch.cuda.empty_cache()

        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics["fps"] = num_samples / total_time  # 计算FPS
        return avg_metrics

    def _calculate_iou(self, y_true, y_pred):
        """交并比计算"""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union if union != 0 else 0

    def _visualize_sample(self, img_tensor, true_mask, pred_mask, batch_idx):
        """可视化结果保存"""
        # 确保保存目录存在
        output_dir = "val_results"
        os.makedirs(output_dir, exist_ok=True)  # 创建目录

        orig_img = self._denormalize(img_tensor)
        pred_np = pred_mask.squeeze()  # 确保预测是二维数组

        processed_mask = self._postprocess_mask(pred_np)
        contours = self._find_contours(processed_mask)

        output_img = orig_img.copy()
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            cv2.drawContours(output_img, [np.intp(box)], 0, (0, 255, 0), 2)

        fig = plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1).imshow(orig_img)  # 确保显示 RGB 格式
        plt.title("Input Image")
        plt.subplot(1, 3, 2).imshow(pred_np, cmap='gray')  # 替换为模型预测掩码
        plt.title("Model Prediction Mask")
        plt.subplot(1, 3, 3).imshow(output_img)  # 确保显示 RGB 格式
        plt.title("Detection Result")
        plt.savefig(f"{output_dir}/batch_{batch_idx:04d}.png")  # 保存到指定目录
        plt.close(fig)
        del fig


if __name__ == "__main__":
    validator = ModelValidator(Config.model_path)
    metrics = validator.evaluate()

    print("\n验证结果（IoU优先指标）:")
    print(f"IoU: {metrics['iou']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f} | mAP: {metrics['ap']:.4f} | FPS: {metrics['fps']:.2f}")