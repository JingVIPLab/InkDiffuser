# -*- coding: utf-8 -*-
import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

'''
def save_single_image(save_dir, image, timestamp):
    """Save a single image with timestamp"""
    save_path = f"{save_dir}/out_single_{timestamp}.jpg"
    image.save(save_path,quality=95)

def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, 
                                style_image_path, resolution, timestamp):
    """Save comparison image with timestamp"""
    new_image = Image.new('RGB', (resolution*3, resolution))
    
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution*2, 0))

    save_path = f"{save_dir}/out_with_cs_{timestamp}.jpg"
    new_image.save(save_path,quality=95)


'''
def save_single_image(save_dir, image, timestamp, char_name=None):
    """Save a single image"""
    save_path = f"{save_dir}/single.jpg"
    image.save(save_path, quality=95)
    return save_path

def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, 
                                style_image_path, resolution, timestamp, char_name=None):
    """Save comparison image"""
    new_image = Image.new('RGB', (resolution*3, resolution))
    
    if content_image_pil is not None:
        content_image = content_image_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    generated_image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(generated_image, (resolution*2, 0))

    save_path = f"{save_dir}/comparison.jpg"
    new_image.save(save_path, quality=95)
    return save_path
    

'''
def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, 
                                style_image_path, resolution, timestamp):
    """Save comparison image with timestamp"""
    new_image = Image.new('RGB', (resolution*3, resolution))
    
    # 处理内容图像
    if content_image_pil is not None:
        content_image = content_image_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    # 处理样式图像
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    # 处理生成的图像
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    generated_image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)

    # 粘贴图像
    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(generated_image, (resolution*2, 0))

    # 保存图像
    save_path = f"{save_dir}/out_with_cs_{timestamp}.jpg"
    new_image.save(save_path, quality=95)
    
    return save_path

'''
    
def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon with numerical stability checks
    """
    # 首先检查输入是否包含NaN或Inf
    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        print(f"警告: noise_pred包含NaN或Inf值，将被清理")
        noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(x_t).any() or torch.isinf(x_t).any():
        print(f"警告: x_t包含NaN或Inf值，将被清理")
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        try:
            pred_original_sample_i = scheduler.step(
                model_output=noise_pred_i,
                timestep=t,
                sample=x_t_i,
                # predict_epsilon=True,
                generator=None,
                return_dict=True,
            ).pred_original_sample
            
            # 数值稳定性检查：清理任何NaN或Inf值
            pred_original_sample_i = torch.nan_to_num(
                pred_original_sample_i, 
                nan=0.0, 
                posinf=10.0,  # 限制正无穷值
                neginf=-10.0  # 限制负无穷值
            )
            
            # 额外的数值范围限制，防止极端值
            pred_original_sample_i = torch.clamp(pred_original_sample_i, min=-50.0, max=50.0)
            
        except Exception as e:
            print(f"警告: scheduler.step在第{i}个样本时出错: {e}，使用零填充")
            pred_original_sample_i = torch.zeros_like(x_t_i)
        
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)

    return pred_original_sample


def reNormalize_img(pred_original_sample):
    # 数值稳定性检查：清理输入中的NaN或Inf值
    if torch.isnan(pred_original_sample).any() or torch.isinf(pred_original_sample).any():
        print(f"警告: reNormalize_img输入包含NaN或Inf值，将被清理")
        pred_original_sample = torch.nan_to_num(pred_original_sample, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 额外的范围限制，防止极端值影响后续计算
    pred_original_sample = torch.clamp(pred_original_sample, min=-100.0, max=100.0)
    
    # 安全的重新归一化
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    
    # 最终检查：确保输出没有NaN或Inf
    pred_original_sample = torch.nan_to_num(pred_original_sample, nan=0.5, posinf=1.0, neginf=0.0)
    
    return pred_original_sample


def normalize_mean_std(image):
    # 数值稳定性检查：清理输入中的NaN或Inf值
    if torch.isnan(image).any() or torch.isinf(image).any():
        print(f"警告: normalize_mean_std输入包含NaN或Inf值，将被清理")
        image = torch.nan_to_num(image, nan=0.5, posinf=1.0, neginf=0.0)
    
    # 确保输入在合理范围内
    image = torch.clamp(image, min=0.0, max=1.0)
    
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    try:
        image = transforms_norm(image)
    except Exception as e:
        print(f"警告: normalize_mean_std变换出错: {e}，使用安全默认值")
        # 如果变换失败，手动进行归一化
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)) / \
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
    
    # 最终检查：确保输出没有NaN或Inf
    image = torch.nan_to_num(image, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return image


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128):
    pygame.init()

    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, fsize=128):
    
    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    bg = np.full((fsize, fsize), 255)
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    im = copy.deepcopy(bg)
    h, w = imo.shape[:2]
    if h > fsize:
        h, w = fsize, round(w*fsize/h)
        imo = cv2.resize(imo, (w, h))
    if w > fsize:
        h, w = round(h*fsize/w), fsize
        imo = cv2.resize(imo, (w, h))
    x, y = round((fsize-w)/2), round((fsize-h)/2)
    im[y:h+y, x:x+w] = imo
    pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
    
    return pil_im


def comprehensive_numerical_check(tensor_dict, step_name="", log_func=print):
    """
    综合的数值稳定性检查工具
    
    Args:
        tensor_dict: 包含张量的字典，键为张量名称，值为张量
        step_name: 当前检查的步骤名称
        log_func: 日志记录函数
    
    Returns:
        dict: 包含检查结果的字典
    """
    results = {
        'has_nan': False,
        'has_inf': False,
        'has_extreme_values': False,
        'problematic_tensors': [],
        'statistics': {}
    }
    
    for name, tensor in tensor_dict.items():
        if not torch.is_tensor(tensor):
            continue
            
        # 基本数值检查
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan:
            results['has_nan'] = True
            results['problematic_tensors'].append(f"{name}(NaN)")
            nan_count = torch.isnan(tensor).sum().item()
            log_func(f"🚨 {step_name} - {name}: 发现 {nan_count} 个NaN值")
        
        if has_inf:
            results['has_inf'] = True
            results['problematic_tensors'].append(f"{name}(Inf)")
            inf_count = torch.isinf(tensor).sum().item()
            log_func(f"🚨 {step_name} - {name}: 发现 {inf_count} 个Inf值")
        
        # 统计信息
        try:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            results['statistics'][name] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'shape': list(tensor.shape)
            }
            
            # 检查极值
            if abs(max_val) > 1e6 or abs(min_val) > 1e6:
                results['has_extreme_values'] = True
                results['problematic_tensors'].append(f"{name}(极值)")
                log_func(f"⚠️  {step_name} - {name}: 极值范围 [{min_val:.2e}, {max_val:.2e}]")
            
            # 检查梯度爆炸/消失
            if abs(mean_val) > 100 or abs(std_val) > 100:
                log_func(f"⚠️  {step_name} - {name}: 可能的梯度问题 - mean: {mean_val:.2e}, std: {std_val:.2e}")
            
            if abs(mean_val) < 1e-7 and abs(std_val) < 1e-7:
                log_func(f"⚠️  {step_name} - {name}: 可能的梯度消失 - mean: {mean_val:.2e}, std: {std_val:.2e}")
                
        except Exception as e:
            log_func(f"❌ {step_name} - {name}: 统计计算失败: {e}")
    
    return results


def safe_optimizer_step(optimizer, model, max_grad_norm=1.0, log_func=print):
    """
    安全的优化器步骤，包含梯度检查和裁剪
    
    Args:
        optimizer: 优化器
        model: 模型
        max_grad_norm: 最大梯度范数
        log_func: 日志函数
    
    Returns:
        bool: 是否成功执行优化步骤
    """
    try:
        # 检查梯度
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[f"grad_{name}"] = param.grad
        
        if grad_dict:
            grad_check = comprehensive_numerical_check(grad_dict, "梯度检查", log_func)
            
            if grad_check['has_nan'] or grad_check['has_inf']:
                log_func("🚨 检测到梯度中有NaN或Inf，跳过优化步骤")
                optimizer.zero_grad()
                return False
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            log_func("🚨 梯度范数为NaN或Inf，跳过优化步骤")
            optimizer.zero_grad()
            return False
        
        if grad_norm > max_grad_norm * 10:  # 极端梯度爆炸
            log_func(f"⚠️  检测到极端梯度爆炸: {grad_norm:.2e}")
        
        # 执行优化步骤
        optimizer.step()
        return True
        
    except Exception as e:
        log_func(f"❌ 优化器步骤执行失败: {e}")
        return False


def create_numerical_stability_config():
    """
    创建数值稳定性配置建议
    
    Returns:
        dict: 包含建议配置的字典
    """
    config = {
        "mixed_precision": {
            "recommended": "no",  # 对于数值敏感的任务建议关闭
            "alternative": "bf16",  # 如果必须使用，bf16比fp16更稳定
            "reason": "fp16容易导致数值下溢和梯度爆炸"
        },
        "optimizer": {
            "adam_epsilon": 1e-6,  # 增加epsilon提高稳定性
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "weight_decay": 1e-2,
            "max_grad_norm": 1.0
        },
        "learning_rate": {
            "initial": 1e-4,
            "warmup_steps": 10000,
            "scheduler": "linear",
            "note": "避免过高的学习率导致数值不稳定"
        },
        "loss_function": {
            "perceptual_coefficient": 1.0,
            "offset_coefficient": 0.5,
            "dis_loss_weight": 0.001,
            "note": "避免损失权重过大导致训练不稳定"
        }
    }
    return config
