import os
import time  # 用于计算推理时间
import torch
from PIL import Image
from matplotlib import rcParams
from torchvision import transforms
from CRNN import CRNN
from tqdm import tqdm
from difflib import SequenceMatcher
"""
评估网络模型性能代码
"""

# -------------------- 字符到索引的映射 --------------------
CHARS = "京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵青藏川宁琼粤0123456789ABCDEFGHJKLMNPQRSTUVWXYZ挂学警港澳使领"
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
char_to_idx["<blank>"] = 0  # CTC空白符
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
# 设置中文字体，确保 matplotlib 支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 将字体设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------- 数据预处理 --------------------
def preprocess_image(image_path, img_height=32, img_width=160):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Grayscale(num_output_channels=3),  # 将灰度图扩展为3通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 按照3通道计算归一化
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # 增加batch维度

# -------------------- 解码预测结果 --------------------
def decode_predictions(predictions, idx_to_char):
    decoded = []
    prev_char_idx = None
    for char_idx in predictions:
        if char_idx != prev_char_idx and char_idx != 0:
            decoded.append(idx_to_char[char_idx])
        prev_char_idx = char_idx
    return ''.join(decoded)

# -------------------- 计算编辑距离 --------------------
def calculate_edit_distance(predicted, target):
    """
    计算两个字符串之间的编辑距离
    :param predicted: 模型预测的字符串
    :param target: 真实标签字符串
    :return: 编辑距离
    """
    return 1 - SequenceMatcher(None, predicted, target).ratio()

# -------------------- 评估模型性能 --------------------
def evaluate_model(model, test_data, image_dir, device):
    """
    在测试数据集上评估模型性能
    :param model: 加载的模型
    :param test_data: 测试数据集，格式 [(filename, label), ...]
    :param image_dir: 图片所在的目录
    :param device: 设备（CPU 或 GPU）
    :return: 准确率、平均编辑距离和 FPS
    """
    total_samples = len(test_data)
    if total_samples == 0:
        print("测试数据集为空，请检查测试数据文件和格式是否正确！")
        return 0.0, float('inf'), 0.0  # 返回默认值以避免后续计算

    correct_predictions = 0
    total_edit_distance = 0.0
    total_inference_time = 0.0  # 用于记录总推理时间

    model.eval()
    with torch.no_grad():
        for filename, label in tqdm(test_data, desc="Evaluating"):
            image_path = os.path.join(image_dir, filename)  # 拼接图片路径
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                continue

            # 加载和预处理图片
            image_tensor = preprocess_image(image_path).to(device)

            # 开始计时
            start_time = time.time()

            # 模型预测
            logits = model(image_tensor)  # [W, B, C]
            log_probs = logits.log_softmax(2)  # 转换为log概率
            predictions = torch.argmax(log_probs, dim=2).squeeze(1).cpu().numpy()

            # 结束计时
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            # 解码预测结果
            predicted_label = decode_predictions(predictions, idx_to_char)

            # 判断是否预测正确
            if predicted_label == label:
                correct_predictions += 1

            # 计算编辑距离
            total_edit_distance += calculate_edit_distance(predicted_label, label)

    accuracy = correct_predictions / total_samples
    avg_edit_distance = total_edit_distance / total_samples
    fps = total_samples / total_inference_time  # 计算 FPS
    return accuracy, avg_edit_distance, fps

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 配置路径
    model_path = "./checkpoints/best_model.pth"
    test_data_file = "../data/results/locate_plates/spllit/CCPD2020/ccpd_green_test.txt"  # 测试数据集文件，格式为每行 "filename,label"
    image_dir = "../data/results/locate_plates/CCPD2020/ccpd_green_test"  # 图片所在的目录

    # 加载测试数据
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"测试数据集文件不存在: {test_data_file}")

    test_data = []
    with open(test_data_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) == 2:
                test_data.append((parts[0], parts[1]))  # 添加 (filename, label) 元组

    # 检查模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_chars=len(char_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 评估模型
    print("开始评估模型性能...")
    accuracy, avg_edit_distance, fps = evaluate_model(model, test_data, image_dir, device)
    print(f"模型评估完成！")
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"平均编辑距离: {avg_edit_distance:.4f}")
    print(f"FPS（每秒处理帧数）: {fps:.2f}")