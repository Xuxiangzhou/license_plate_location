import os
import random
import shutil


def split_and_copy_files(src_folder, train_folder, val_folder, num_files, train_ratio=0.8):
    """
    从源文件夹中提取一定数量的文件，并按照比例划分到 train 和 val 文件夹

    :param src_folder: 源文件夹路径
    :param train_folder: 训练集目标文件夹路径
    :param val_folder: 验证集目标文件夹路径
    :param num_files: 要提取的文件总数
    :param train_ratio: 训练集比例（默认 0.8，即 80%）
    """
    # 创建目标文件夹
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 获取所有文件名
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    if len(all_files) < num_files:
        raise ValueError(f"源文件夹中文件数量不足，仅有 {len(all_files)} 个文件，但需要 {num_files} 个文件。")

    # 随机选择指定数量的文件
    selected_files = random.sample(all_files, num_files)

    # 按比例划分
    train_count = int(num_files * train_ratio)
    train_files = selected_files[:train_count]
    val_files = selected_files[train_count:]

    # 复制文件到 train 文件夹
    for idx, file in enumerate(train_files):
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(train_folder, file)
        shutil.copy2(src_path, dst_path)  # 保留元数据
        if idx % 100 == 0:
            print(f"已复制到 train 文件夹 {idx + 1}/{len(train_files)}")

    # 复制文件到 val 文件夹
    for idx, file in enumerate(val_files):
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(val_folder, file)
        shutil.copy2(src_path, dst_path)  # 保留元数据
        if idx % 100 == 0:
            print(f"已复制到 val 文件夹 {idx + 1}/{len(val_files)}")

    print(f"完成！训练集文件数：{len(train_files)}，验证集文件数：{len(val_files)}")


#
split_and_copy_files(
    src_folder="../data/val/image", #原图片目录
    train_folder="./Locate/train",  #训练数据集目录
    val_folder="./Locate/train",    #验证数据集目录
    num_files=10000,                  #提取数量，必须小于等于原目录中文件数量
    train_ratio=0.8                   #提取比例，此为train目录所占比例
)