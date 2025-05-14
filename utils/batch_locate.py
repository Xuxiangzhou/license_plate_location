import os
import cv2
from multiprocessing import Pool, cpu_count

from debian.debtags import output

from Locate.locate import predict_license_plate  # 替换为您的脚本文件名
""""
本代码主要实现的功能是批量将CCPD数据集中车牌区域图片从原图中提取出来，并打上标签，标签格式：文件名，车牌号码
需要配置的参数有：
    待识别目录：input_directory = "../data/CCPD2020/ccpd_green/test"  # 输入目录路径
    车牌图片保存目录：output_directory = "../data/results/locate_plates/CCPD2020/ccpd_green_test"  # 输出目录路径
    训练模型地址：model_path = "../weights/best_model_green.pth"  # 模型文件路径
    标签文件输出地址：label_file = "../data/results/locate_plates/spllit/CCPD2020/ccpd_green_test.txt"  #车牌号对应标签地址
"""
# config={
#     "input_directory":"../data/CCPD2020/ccpd_green_test",
#     "output_directory":"../data/results/locate_plates/CCPD2020/ccpd_green_test" , # 输出目录路径
#     "model_path" :"" "../weights/best_model_green.pth",  # 模型文件路径
#     "label_file" : "../data/results/locate_plates/spllit/CCPD2020/ccpd_green_test.txt"  #车牌号对应标签地址
# }
# 定义省份、字母、数字数组
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def parse_plate_number(file_name):
    """
    批量裁剪车牌区域图片并将对应标签保存到指定地址
    参数:
        file_name (str): 图像文件名。
    返回:
        plate_number (str): 解析出的车牌号码。
    """
    try:
        # 获取车牌索引部分
        plate_indices = file_name.split('-')[4]
        indices = list(map(int, plate_indices.split('_')))

        # 构造车牌号码
        plate_number = provinces[indices[0]] + alphabets[indices[1]] + ''.join([ads[idx] for idx in indices[2:]])
        return plate_number
    except (IndexError, ValueError) as e:
        print(f"解析文件名 {file_name} 时出错: {e}")
        return None


def process_single_image(args):
    """
    处理单张图片，提取车牌区域并保存。

    参数:
        args (tuple): 包括输入图片路径、输出目录、模型路径、设备和结果列表。
    """
    image_path, output_dir, model_path, device = args
    image_name = os.path.basename(image_path)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        return None

    # 调用车牌定位函数
    try:
        _, cropped_plate = predict_license_plate(model_path, image, device)
    except Exception as e:
        print(f"车牌定位失败: {image_path}, 错误: {e}")
        return None

    # 如果检测到车牌，保存裁剪后的图片并记录车牌号码
    plate_number = parse_plate_number(image_name)
    if cropped_plate is not None and plate_number:
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, cropped_plate)
        print(f"保存裁剪车牌图片: {output_path}")
        return image_name, plate_number
    else:
        print(f"未检测到车牌或解析失败: {image_path}")
        return None


def batch_process_images(input_dir, output_dir, model_path, device='cuda'):
    """
    使用多进程批量处理图片，提取车牌区域并保存，并生成车牌号码标签文件。

    参数:
        input_dir (str): 输入图片目录路径。
        output_dir (str): 输出图片目录路径。
        model_path (str): 已训练模型的路径。
        device (str): 使用的设备 ('cuda' 或 'cpu')。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有输入图片路径
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(('.jpg', '.png'))]

    # 准备任务参数
    tasks = [(image_path, output_dir, model_path, device) for image_path in image_paths]

    # 使用多进程池处理图片
    with Pool(processes=cpu_count() // 2) as pool:
        results = pool.map(process_single_image, tasks)

    # 过滤掉处理失败的结果
    results = [res for res in results if res is not None]

    # 将车牌号码写入文件
    label_file = "../data/results/locate_plates/spllit/CCPD2020/ccpd_green_test.txt"  #车牌号对应标签地址
    with open(label_file, 'w', encoding='utf-8') as f:
        for image_name, plate_number in results:
            f.write(f"{image_name},{plate_number}\n")  # 使用逗号分隔文件名和车牌号码
    print(f"车牌号码已保存到: {label_file}")


# 示例用法
if __name__ == "__main__":
    input_directory = "../data/CCPD2020/ccpd_green/test"  # 输入目录路径
    output_directory = "../data/results/locate_plates/CCPD2020/ccpd_green_test"  # 输出目录路径
    model_path = "../weights/best_model_green.pth"  # 模型文件路径

    batch_process_images(input_directory, output_directory, model_path)