import os
import argparse
import numpy as np
import cv2
import torch
import lpips
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim_loss
from pytorch_fid import fid_score
import torchvision.transforms as transforms
from PIL import Image
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Image Quality Metrics')
    parser.add_argument('--input_images_path', default='',
                      help='Path to ground truth images')
    parser.add_argument('--generated_images_path', default='',
                      help='Path to generated images')
    parser.add_argument('--image_size', type=int, default=96,
                      help='Size of images for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for evaluation')
    return parser.parse_args()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".JPEG", ".PNG", ".JPG"])

def load_image(filepath, image_size):
    """Load and preprocess image"""
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img = np.array(img) / 255.0
            return img
    except Exception as e:
        print(f"Error loading image {filepath}: {str(e)}")
        return None

class ImagePairDataset(Dataset):
    def __init__(self, gt_dir, gen_dir, image_size):
        super(ImagePairDataset, self).__init__()

        gt_subdirs = [d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))]
        gen_subdirs = [d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))]

        common_subdirs = sorted(list(set(gt_subdirs) & set(gen_subdirs)))
        
        print(f"Found {len(gt_subdirs)} subdirectories in ground truth directory")
        print(f"Found {len(gen_subdirs)} subdirectories in generated directory")
        print(f"Found {len(common_subdirs)} matching subdirectories")
        
        if not common_subdirs:
            raise ValueError("No matching subdirectories found!")
        
        self.gt_paths = []
        self.gen_paths = []
        
        for subdir in common_subdirs:
            gt_subdir_path = os.path.join(gt_dir, subdir)
            gen_subdir_path = os.path.join(gen_dir, subdir)

            gt_files = {f for f in os.listdir(gt_subdir_path) if is_image_file(f)}
            gen_files = {f for f in os.listdir(gen_subdir_path) if is_image_file(f)}

            common_files = sorted(list(gt_files & gen_files))
            
            print(f"\nProcessing subdirectory: {subdir}")
            print(f"Found {len(gt_files)} files in ground truth")
            print(f"Found {len(gen_files)} files in generated")
            print(f"Found {len(common_files)} matching pairs")
            
            # 添加匹配的图像对
            for filename in common_files:
                self.gt_paths.append(os.path.join(gt_subdir_path, filename))
                self.gen_paths.append(os.path.join(gen_subdir_path, filename))
        
        if not self.gt_paths:
            raise ValueError("No matching image pairs found in any subdirectory!")
        
        print(f"\nTotal matching image pairs found: {len(self.gt_paths)}")
        self.image_size = image_size
        
        # 验证所有图片是否可以正确加载
        self.validate_images()

    def validate_images(self):
        """验证所有图片对是否可以正确加载"""
        valid_pairs = []
        for gt_path, gen_path in tqdm(zip(self.gt_paths, self.gen_paths), 
                                    desc="Validating images", 
                                    total=len(self.gt_paths)):
            gt_img = load_image(gt_path, self.image_size)
            gen_img = load_image(gen_path, self.image_size)
            
            if gt_img is not None and gen_img is not None:
                valid_pairs.append((gt_path, gen_path))
            else:
                print(f"Skipping invalid pair: {gt_path} - {gen_path}")
        
        if valid_pairs:
            self.gt_paths, self.gen_paths = zip(*valid_pairs)
            self.gt_paths = list(self.gt_paths)
            self.gen_paths = list(self.gen_paths)
        else:
            self.gt_paths = []
            self.gen_paths = []
            
        print(f"Valid pairs after validation: {len(self.gt_paths)}")

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        gen_path = self.gen_paths[index]
        
        gt_img = load_image(gt_path, self.image_size)
        gen_img = load_image(gen_path, self.image_size)
        
        # 转换为PyTorch张量
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float()
        gen_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).float()
        
        return gt_tensor, gen_tensor, gt_path, gen_path

def calculate_metrics(args):
    """计算所有评估指标"""
    print("\nInitializing evaluation...")
    
    # 初始化数据集和加载器
    dataset = ImagePairDataset(args.input_images_path, args.generated_images_path, args.image_size)
    dataloader = DataLoader(dataset, 
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=(args.device != 'cpu'))
    
    # 初始化LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').to(args.device)
    
    # 初始化指标存储
    metrics = {
        'ssim': [],
        'lpips': [],
        'l1': [],
        'rmse': [],
        'psnr': [],
        'processed_pairs': [],
        'subdir_metrics': {}
    }
    
    # 计算每对图像的指标
    print("\nCalculating metrics for each image pair...")
    for gt_batch, gen_batch, gt_paths, gen_paths in tqdm(dataloader):
        # 移动数据到设备
        gt_batch = gt_batch.to(args.device)
        gen_batch = gen_batch.to(args.device)
        
        # 转换为numpy计算其他指标
        gt_np = gt_batch.cpu().numpy().transpose(0, 2, 3, 1)
        gen_np = gen_batch.cpu().numpy().transpose(0, 2, 3, 1)
        
        # 对批次中的每张图片分别计算所有指标
        for i in range(gt_np.shape[0]):
            # 获取单张图片的tensor
            gt_single = gt_batch[i:i+1]
            gen_single = gen_batch[i:i+1]
            
            # 计算LPIPS（针对单张图片）
            lpips_value = loss_fn_alex(gt_single, gen_single).item()
            metrics['lpips'].append(lpips_value)
            
            # 计算L1（针对单张图片）
            l1_value = torch.mean(torch.abs(gt_single - gen_single)).item()
            metrics['l1'].append(l1_value)
            
            # 计算SSIM
            ssim = ssim_loss(gt_np[i], gen_np[i],
                           data_range=1.0,
                           channel_axis=2)
            metrics['ssim'].append(ssim)
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((gt_np[i] - gen_np[i]) ** 2))
            metrics['rmse'].append(rmse)
            
            # 计算PSNR（添加数值稳定性检查）
            mse = np.mean((gt_np[i] - gen_np[i]) ** 2)
            if mse == 0:
                psnr = float('inf')  # 完全相同的图像
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            metrics['psnr'].append(psnr)
            
            # 记录处理的图片对
            metrics['processed_pairs'].append((gt_paths[i], gen_paths[i]))
            
            # 获取子文件夹名称并记录指标
            subdir = os.path.basename(os.path.dirname(gt_paths[i]))
            if subdir not in metrics['subdir_metrics']:
                metrics['subdir_metrics'][subdir] = {
                    'ssim': [], 'lpips': [], 'l1': [],
                    'rmse': [], 'psnr': []
                }
            metrics['subdir_metrics'][subdir]['ssim'].append(ssim)
            metrics['subdir_metrics'][subdir]['lpips'].append(lpips_value)
            metrics['subdir_metrics'][subdir]['l1'].append(l1_value)
            metrics['subdir_metrics'][subdir]['rmse'].append(rmse)
            metrics['subdir_metrics'][subdir]['psnr'].append(psnr)
    
    # 计算FID
    print("\nCalculating FID score...")
    try:
        # 创建临时目录来存储所有图像
        temp_gt_dir = os.path.join('evaluation_results', 'temp_gt')
        temp_gen_dir = os.path.join('evaluation_results', 'temp_gen')
        os.makedirs(temp_gt_dir, exist_ok=True)
        os.makedirs(temp_gen_dir, exist_ok=True)
        
        # 复制所有图像到临时目录，使用唯一文件名避免冲突
        for idx, (gt_path, gen_path) in enumerate(zip(dataset.gt_paths, dataset.gen_paths)):
            # 获取子文件夹名和原文件名
            subdir = os.path.basename(os.path.dirname(gt_path))
            gt_basename = os.path.basename(gt_path)
            gen_basename = os.path.basename(gen_path)
            
            # 创建唯一文件名
            gt_unique_name = f"{subdir}_{gt_basename}"
            gen_unique_name = f"{subdir}_{gen_basename}"
            
            # 如果仍有重复，添加索引
            gt_final_path = os.path.join(temp_gt_dir, gt_unique_name)
            gen_final_path = os.path.join(temp_gen_dir, gen_unique_name)
            
            if os.path.exists(gt_final_path):
                name, ext = os.path.splitext(gt_unique_name)
                gt_unique_name = f"{name}_{idx}{ext}"
                gt_final_path = os.path.join(temp_gt_dir, gt_unique_name)
                
            if os.path.exists(gen_final_path):
                name, ext = os.path.splitext(gen_unique_name)
                gen_unique_name = f"{name}_{idx}{ext}"
                gen_final_path = os.path.join(temp_gen_dir, gen_unique_name)
            
            # 复制文件
            shutil.copy2(gt_path, gt_final_path)
            shutil.copy2(gen_path, gen_final_path)
        
        # 计算FID
        fid_value = fid_score.calculate_fid_given_paths(
            [temp_gt_dir, temp_gen_dir],
            args.batch_size,
            args.device,
            dims=2048,
            num_workers=args.num_workers
        )
        
        # 清理临时目录
        shutil.rmtree(temp_gt_dir)
        shutil.rmtree(temp_gen_dir)
        
    except Exception as e:
        print(f"Warning: FID calculation failed: {str(e)}")
        fid_value = float('inf')  # 设置一个默认值
    
    # 移除OCR字体识别指标相关代码
    
    # 计算平均值（处理PSNR中的无穷大值）
    psnr_values = [p for p in metrics['psnr'] if p != float('inf')]
    psnr_mean = np.mean(psnr_values) if psnr_values else float('inf')
    
    results = {
        'FID': fid_value,
        'LPIPS': np.mean(metrics['lpips']),
        'SSIM': np.mean(metrics['ssim']),
        'L1': np.mean(metrics['l1']),
        'RMSE': np.mean(metrics['rmse']),
        'PSNR': psnr_mean,
        'Total_Images': len(metrics['processed_pairs']),
        'Subdir_Results': {}
    }
    
    # 计算每个子文件夹的平均指标
    for subdir, subdir_metrics in metrics['subdir_metrics'].items():
        subdir_psnr_values = [p for p in subdir_metrics['psnr'] if p != float('inf')]
        subdir_psnr_mean = np.mean(subdir_psnr_values) if subdir_psnr_values else float('inf')
        
        results['Subdir_Results'][subdir] = {
            'SSIM': np.mean(subdir_metrics['ssim']),
            'LPIPS': np.mean(subdir_metrics['lpips']),
            'L1': np.mean(subdir_metrics['l1']),
            'RMSE': np.mean(subdir_metrics['rmse']),
            'PSNR': subdir_psnr_mean,
            'Total_Images': len(subdir_metrics['ssim'])
        }
    
    return results, metrics

def save_results(results, metrics, args):
    """保存评估结果"""
    # 创建结果目录
    os.makedirs('evaluation_results', exist_ok=True)
    
    # 保存总体指标
    with open('evaluation_results/summary.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Total Images Processed: {results['Total_Images']}\n")
        f.write(f"FID Score: {results['FID']:.4f}\n")
        f.write(f"LPIPS Score: {results['LPIPS']:.4f}\n")
        f.write(f"SSIM Score: {results['SSIM']:.4f}\n")
        f.write(f"L1 Score: {results['L1']:.4f}\n")
        f.write(f"RMSE Score: {results['RMSE']:.4f}\n")
        
        # 处理PSNR的显示
        if results['PSNR'] == float('inf'):
            f.write(f"PSNR Score: Inf (perfect match)\n")
        else:
            f.write(f"PSNR Score: {results['PSNR']:.4f}\n")
        
        # 移除OCR结果保存
        f.write("\n")
        
        # 保存每个子文件夹的结果
        f.write("Results by Subdirectory:\n")
        for subdir, subdir_results in results['Subdir_Results'].items():
            f.write(f"\n{subdir}:\n")
            f.write(f"  Total Images: {subdir_results['Total_Images']}\n")
            f.write(f"  SSIM Score: {subdir_results['SSIM']:.4f}\n")
            f.write(f"  LPIPS Score: {subdir_results['LPIPS']:.4f}\n")
            f.write(f"  L1 Score: {subdir_results['L1']:.4f}\n")
            f.write(f"  RMSE Score: {subdir_results['RMSE']:.4f}\n")
            
            # 处理子目录PSNR的显示
            if subdir_results['PSNR'] == float('inf'):
                f.write(f"  PSNR Score: Inf (perfect match)\n")
            else:
                f.write(f"  PSNR Score: {subdir_results['PSNR']:.4f}\n")
    
    # 保存每对图片的详细指标
    with open('evaluation_results/detailed_metrics.txt', 'w') as f:
        f.write("Per-Image Metrics:\n")
        for i, (gt_path, gen_path) in enumerate(metrics['processed_pairs']):
            f.write(f"\nImage Pair {i+1}:\n")
            f.write(f"Ground Truth: {gt_path}\n")
            f.write(f"Generated: {gen_path}\n")
            f.write(f"SSIM: {metrics['ssim'][i]:.4f}\n")
            f.write(f"LPIPS: {metrics['lpips'][i]:.4f}\n")
            f.write(f"L1: {metrics['l1'][i]:.4f}\n")
            f.write(f"RMSE: {metrics['rmse'][i]:.4f}\n")
            
            # 处理PSNR的显示
            if metrics['psnr'][i] == float('inf'):
                f.write(f"PSNR: Inf (perfect match)\n")
            else:
                f.write(f"PSNR: {metrics['psnr'][i]:.4f}\n")

def main():
    # 解析参数
    args = parse_args()
    
    # 打印评估配置
    print("Evaluation Configuration:")
    print(f"Ground Truth Path: {args.input_images_path}")
    print(f"Generated Images Path: {args.generated_images_path}")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    # 移除OCR相关打印
    
    try:
        # 计算指标
        results, metrics = calculate_metrics(args)
        
        # 打印结果
        print("\nEvaluation Results:")
        print(f"Total Images Processed: {results['Total_Images']}")
        print(f"FID Score: {results['FID']:.4f}")
        print(f"LPIPS Score: {results['LPIPS']:.4f}")
        print(f"SSIM Score: {results['SSIM']:.4f}")
        print(f"L1 Score: {results['L1']:.4f}")
        print(f"RMSE Score: {results['RMSE']:.4f}")
        
        # 处理PSNR的显示
        if results['PSNR'] == float('inf'):
            print(f"PSNR Score: Inf (perfect match)")
        else:
            print(f"PSNR Score: {results['PSNR']:.4f}")
        
        # 移除OCR结果打印
        
        # 保存结果
        save_results(results, metrics, args)
        print("\nResults have been saved to 'evaluation_results' directory")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    main()