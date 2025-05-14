import os

# 定义省份、字母、数字数组
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
# 这个代码主要用于从车牌区域图片中解析该车牌对应的车牌号码信息，batch_locate.py中已经集成，无需单独运行
def parse_plate_number(file_name):
    """
    从文件名中解析车牌号码。
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


def parse_license_plates(input_dir, output_file):
    """
    解析指定目录的文件名，生成车牌号码，并保存到txt文件。

    参数:
        input_dir (str): 输入目录路径。
        output_file (str): 输出txt文件路径。
    """
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    # 获取目录下的所有文件
    file_names = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    # 解析文件名并保存车牌号码
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in file_names:
            plate_number = parse_plate_number(file_name)
            if plate_number:
                f.write(f"{file_name},{plate_number}\n")
                print(f"文件: {file_name} -> 车牌号码: {plate_number}")
            else:
                print(f"跳过文件: {file_name}")

    print(f"车牌号码已保存到: {output_file}")


# 示例用法
if __name__ == "__main__":
    input_directory = "data/CCPD2020/ccpd_green/train"  # 替换为您的输入目录路径
    output_txt = "data/results/locate_plates/spllit/ccpd_green_train.txt"  # 替换为您的输出txt文件路径

    parse_license_plates(input_directory, output_txt)