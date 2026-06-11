import os
import torch
import numpy as np
from PIL import Image
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def sobel_filter(img):
    """使用Sobel算子提取图像的高频信息
    Args:
        img: [B, C, H, W] 输入图像
    Returns:
        高频特征 [B, 1, H, W] - 单通道高频特征
    """
    # 确保输入是正确的格式
    if len(img.shape) != 4:
        raise ValueError(f"Input should be [B, C, H, W], got {img.shape}")
    
    # 首先将图像转换为灰度
    if img.shape[1] == 3:  # RGB图像
        # 使用RGB到灰度的标准转换公式
        gray_img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    else:
        # 如果已经是单通道，直接使用
        gray_img = img
    
    # 定义Sobel卷积核
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                           dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                           dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    
    # 应用Sobel算子
    pad = F.pad(gray_img, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(pad, sobel_x)
    grad_y = F.conv2d(pad, sobel_y)
    
    # 计算梯度幅值，添加数值稳定性
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 添加小常数防止sqrt(0)
    
    # 安全的归一化到[-1, 1]范围
    grad_min = torch.min(grad_magnitude)
    grad_max = torch.max(grad_magnitude)
    
    # 检查是否有有效的梯度变化
    if torch.abs(grad_max - grad_min) < 1e-8:
        # 如果没有梯度变化，返回零矩阵
        grad_magnitude = torch.zeros_like(grad_magnitude)
    else:
        # 安全归一化
        grad_magnitude = (grad_magnitude - grad_min) / (grad_max - grad_min + 1e-8) * 2 - 1
    
    # 数值稳定性检查：确保没有NaN或Inf
    grad_magnitude = torch.nan_to_num(grad_magnitude, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 返回单通道结果
    return grad_magnitude

def extract_high_freq(input_path, output_dir, device="cpu"):
    """提取图像的高频信息并保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不包含扩展名）用于输出
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 加载并预处理图像
    img = Image.open(input_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # 提取高频信息
    with torch.no_grad():
        high_freq = sobel_filter(img_tensor)  # [1, 1, H, W]
    
    # 将结果转换为PIL图像
    # 首先将[-1, 1]范围转换为[0, 1]
    high_freq = (high_freq + 1) / 2
    high_freq_np = high_freq.squeeze().cpu().numpy()  # [H, W]
    
    # 保存图像
    plt.figure(figsize=(10, 10))
    plt.imshow(high_freq_np, cmap='gray')
    plt.axis('off')
    output_path = os.path.join(output_dir, f"{filename}_high_freq.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存彩色热力图版本
    plt.figure(figsize=(10, 10))
    plt.imshow(high_freq_np, cmap='viridis')
    plt.axis('off')
    output_path_color = os.path.join(output_dir, f"{filename}_high_freq_color.png")
    plt.savefig(output_path_color, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存原始图像和高频图像的对比图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(np.array(img))
    axes[0].set_title("原始图像", fontsize=16)
    axes[0].axis('off')
    axes[1].imshow(high_freq_np, cmap='gray')
    axes[1].set_title("高频信息", fontsize=16)
    axes[1].axis('off')
    comparison_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"已保存高频图像到: {output_path}")
    print(f"已保存彩色高频图像到: {output_path_color}")
    print(f"已保存对比图到: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="使用Sobel算子提取图像高频信息")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="./high_freq_outputs", help="输出目录")
    parser.add_argument("--device", type=str, default="cpu", help="使用的设备 (cuda:0, cuda:1, cpu 等)")
    args = parser.parse_args()
    
    extract_high_freq(args.input, args.output_dir, args.device)

if __name__ == "__main__":
    main()