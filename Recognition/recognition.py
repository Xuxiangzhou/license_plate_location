import os
import torch
from PIL import Image
from torchvision import transforms
from .CRNN import CRNN

"""
此代码定义了一个函数，输入一张图片，通过调用CRNN模型进行字符识别，输出车牌书别结果
"""
# -------------------- 字符到索引的映射 --------------------
CHARS = "京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵青藏川宁琼粤0123456789ABCDEFGHJKLMNPQRSTUVWXYZ挂学警港澳使领"
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
char_to_idx["<blank>"] = 0  # CTC空白符
idx_to_char = {idx: char for char, idx in char_to_idx.items()}  # 索引到字符的映射
# -------------------- 数据预处理 --------------------
def preprocess_image(image, img_height=32, img_width=160):
    """
    对输入图像进行预处理
    :param image_path: 图片路径
    :param img_height: 输入图像的高度
    :param img_width: 输入图像的宽度
    :return: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Grayscale(num_output_channels=3),  # 将灰度图扩展为3通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 按照3通道计算归一化
    ])
    return transform(image).unsqueeze(0)  # 增加batch维度

# -------------------- 解码预测结果 --------------------
def decode_predictions(predictions, idx_to_char):
    """
    将模型预测的索引序列解码为实际字符
    :param predictions: 模型预测的索引序列
    :param idx_to_char: 索引到字符的映射字典
    :return: 解码后的字符结果
    """
    decoded = []
    prev_char_idx = None
    for char_idx in predictions:
        # CTC解码：跳过重复字符索引和空白符索引
        if char_idx != prev_char_idx and char_idx != 0:
            decoded.append(idx_to_char[char_idx])
        prev_char_idx = char_idx
    return ''.join(decoded)

# -------------------- 主函数 --------------------
def recognize_license_plate(image, model_path):
    """
    使用训练好的模型识别车牌
    :param image_path: 输入车牌图片路径
    :param model_path: 训练好的模型路径
    :return: 识别出的车牌号
    """
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_chars=len(char_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()

    # 预处理图片
    image_tensor = preprocess_image(image).to(device)

    # 模型预测
    with torch.no_grad():
        logits = model(image_tensor)  # [W, B, C]
        log_probs = logits.log_softmax(2)  # 转换为log概率
        predictions = torch.argmax(log_probs, dim=2).squeeze(1).cpu().numpy()

    # 解码预测结果
    plate_number = decode_predictions(predictions, idx_to_char)  # 传入 idx_to_char
    return plate_number


if __name__ == "__main__":
    # 输入图片路径和模型路径
    image_path = input("请输入图片路径: ")
    model_path = "checkpoints/best_crnn.pth"
    image = Image.open(image_path)
    # 识别车牌
    if not os.path.exists(image_path):
        print(f"图片路径不存在: {image_path}")
    elif not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
    else:
        plate_number = recognize_license_plate(image, model_path)
        print(f"识别结果: {plate_number}")