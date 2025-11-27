import os
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import re
from typing import List, Dict, Tuple, Optional


def extract_rgb_features(image_path: str) -> Tuple[float, float, float]:
    """
    从图像中提取RGB通道的平均像素值

    参数:
    image_path: 图像文件的路径

    返回:
    三元组，包含蓝、绿、红通道的平均像素值
    """
    # 使用OpenCV读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 分离RGB通道（注意：OpenCV读取的图像顺序是BGR而非RGB）
    blue_channel, green_channel, red_channel = cv2.split(image)

    # 计算每个通道的平均像素值
    blue_mean = np.mean(blue_channel)
    green_mean = np.mean(green_channel)
    red_mean = np.mean(red_channel)

    return blue_mean, green_mean, red_mean


def get_concentration_label(file_name: str, pattern: str = r'([A-Z])_') -> Optional[str]:
    """
    从文件名中提取浓度标签

    参数:
    file_name: 文件名
    pattern: 正则表达式模式，用于匹配浓度标签

    返回:
    匹配到的浓度标签，如果没有匹配则返回None
    """
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    return None


def process_images_in_directory(root_directory: str, output_csv: str, pattern: str = r'([A-Z])_') -> None:
    """
    递归处理目录中的所有图像文件，提取RGB特征并保存到CSV文件

    参数:
    root_directory: 包含图像文件的根目录
    output_csv: 输出CSV文件的路径
    pattern: 用于从文件名提取浓度标签的正则表达式模式
    """
    # 存储提取的特征数据
    data = []

    # 图像文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # 递归遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # 检查是否为图像文件
            if Path(filename).suffix.lower() in image_extensions:
                image_path = os.path.join(dirpath, filename)

                try:
                    # 提取RGB特征
                    blue_mean, green_mean, red_mean = extract_rgb_features(image_path)

                    # 从文件名提取浓度标签
                    concentration_label = get_concentration_label(filename, pattern)

                    # 获取相对路径（相对于root_directory）
                    relative_path = os.path.relpath(dirpath, root_directory)

                    # 添加到数据列表
                    data.append({
                        'filename': filename,
                        'folder': relative_path,
                        'Blue': blue_mean,
                        'Green': green_mean,
                        'Red': red_mean,
                        'Concentration_Label': concentration_label
                    })

                    print(
                        f"已处理: {os.path.join(relative_path, filename)}, RGB均值: ({blue_mean:.2f}, {green_mean:.2f}, {red_mean:.2f}), 标签: {concentration_label}")

                except Exception as e:
                    print(f"处理图像 {image_path} 时出错: {str(e)}")

    # 创建DataFrame并保存到CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"\n数据已成功保存到: {output_csv}")
        print(f"处理的图像数量: {len(data)}")
        print("数据格式示例:")
        print(df.head().to_string())
    else:
        print("未找到可处理的图像文件!")


if __name__ == "__main__":
    # 配置参数
    IMAGE_DIR = "C:/Users/ZhuanZ/Desktop/FL-PIC-NDI5/conc"  # 替换为实际图像文件夹路径
    OUTPUT_CSV = "rgb_features.csv"  # 输出CSV文件名称
    LABEL_PATTERN = r'([A-Z])_'  # 正则表达式模式，用于从文件名提取浓度标签

    # 处理图像并生成CSV
    process_images_in_directory(IMAGE_DIR, OUTPUT_CSV, LABEL_PATTERN)    