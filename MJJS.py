from PIL import Image
import cv2
import numpy as np
import os
import sys
from pathlib import Path

def calculate_image_stats(image_path, border_thickness=2):
    """
    计算二值化图像中的黑白像素统计信息
    
    :param image_path: 图像路径
    :param border_thickness: 边框厚度，默认为2像素
    :return: 返回包含统计信息的字典
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("无法读取图像")
    
    # 创建内部区域掩码（排除边框）
    h, w = img.shape
    mask = np.ones((h-2*border_thickness, w-2*border_thickness), dtype=np.uint8)
    inner_img = img[border_thickness:-border_thickness, border_thickness:-border_thickness]
    
    # 计算内部区域的黑白像素
    white_pixels = np.sum(inner_img == 255)
    black_pixels = np.sum(inner_img == 0)
    total_pixels = white_pixels + black_pixels
    
    # 计算比例
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
    
    # 计算内部面积（像素数）
    inner_area = total_pixels
    
    return {
        'inner_area': inner_area,
        'white_pixels': white_pixels,
        'black_pixels': black_pixels,
        'total_pixels': total_pixels,
        'white_ratio': white_ratio,
        'black_ratio': black_ratio
    }

if __name__ == '__main__':
    try:
        # 使用实际存在的测试图像路径
        image_path = r"D:\LDT\scan_2.png"
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 找不到图像文件 '{image_path}'")
            sys.exit(1)
            
        result = calculate_image_stats(image_path)
        
        print("\n图像统计结果：")
        print(f"内部面积（像素数）: {result['inner_area']}")
        print(f"白色像素数量: {result['white_pixels']}")
        print(f"黑色像素数量: {result['black_pixels']}")
        print(f"白色像素占比: {result['white_ratio']:.2%}")
        print(f"黑色像素占比: {result['black_ratio']:.2%}")
            
    except cv2.error as e:
        print(f"OpenCV错误: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")