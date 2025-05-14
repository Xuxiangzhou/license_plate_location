""""
本代码用于生成车牌掩码，通过解析CCPD数据集文件名实现
"""

import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

#生成车牌区域掩码
def parse_filename(filename):
    """解析CCPD文件名中的四个顶点坐标"""
    try:
        parts = filename.split("-")
        vertices_str = parts[3].split("_")
        vertices = []
        for point_str in vertices_str:
            x, y = map(int, point_str.split("&"))
            vertices.append([x, y])
        return np.array([vertices], dtype=np.int32)
    except (IndexError, ValueError) as e:
        print(f"文件名解析失败: {filename} - 错误: {str(e)}")
        return None


def process_single_file(args):
    input_dir, output_dir, filename = args
    try:
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return (filename, False, "无法读取图像")

        # 解析顶点
        vertices = parse_filename(filename)
        if vertices is None:
            return (filename, False, "顶点解析失败")

        # 创建掩码
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, color=255)

        # 保存结果
        cv2.imwrite(output_path, mask)
        return (filename, True, "成功")
    except Exception as e:
        return (filename, False, str(e))


def generate_masks_parallel(input_dir, output_dir):
    # 准备任务列表
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    task_args = [(input_dir, output_dir, f) for f in filenames]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 多进程池配置
    num_workers = min(cpu_count() * 2, 16)  # 限制最大16进程
    results = []

    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        # 使用tqdm包装任务迭代器
        with tqdm(total=len(filenames), desc="🚀 生成掩码", unit="img", dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(process_single_file, task_args):
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    "成功率": f"{sum(r[1] for r in results) / len(results):.1%}",
                    "进程数": num_workers
                })

    # 打印统计信息
    success_count = sum(r[1] for r in results)
    error_log = [r for r in results if not r[1]]
    print(f"\n处理完成: {success_count}/{len(filenames)} 成功")
    if error_log:
        print("\n错误文件列表:")
        for filename, status, msg in error_log[:5]:  # 显示前5个错误
            print(f"- {filename}: {msg}")
        if len(error_log) > 5:
            print(f"... 共 {len(error_log)} 个错误文件")


if __name__ == "__main__":
    # 配置路径
    input_dir = "../data/CCPD2019/ccpd_rotate"  #原图目录，切换为CCPD目录下数据路径即可
    output_dir = "../data/CCPD2019/ccpd_rotate-mask"    #输出目录生成掩码保存的位置
    # 启动多进程处理
    generate_masks_parallel(input_dir, output_dir)