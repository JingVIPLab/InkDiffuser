import torch
import torch.nn as nn
import torchvision 
from .modules.dis_loss import DifferentiableInkStructureLoss, create_dis_loss_preset


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]


class ContentPerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        # 数值稳定性检查 - 输入图像
        if torch.isnan(generated_images).any() or torch.isinf(generated_images).any():
            print("警告: generated_images包含NaN或Inf值，将被清理")
            generated_images = torch.nan_to_num(generated_images, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(target_images).any() or torch.isinf(target_images).any():
            print("警告: target_images包含NaN或Inf值，将被清理")
            target_images = torch.nan_to_num(target_images, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 确保输入在合理范围内
        generated_images = torch.clamp(generated_images, min=-3.0, max=3.0)
        target_images = torch.clamp(target_images, min=-3.0, max=3.0)

        try:
            generated_features = self.VGG(generated_images)
            target_features = self.VGG(target_images)
        except Exception as e:
            print(f"警告: VGG特征提取出错: {e}，返回零损失")
            return torch.tensor(0.0, device=device)

        perceptual_loss = 0
        valid_layers = 0
        
        # 安全地计算每层的损失
        for i in range(min(len(generated_features), len(target_features))):
            try:
                gen_feat = generated_features[i]
                tar_feat = target_features[i]
                
                # 检查特征是否有异常值
                if torch.isnan(gen_feat).any() or torch.isinf(gen_feat).any():
                    print(f"警告: generated_features[{i}]包含NaN或Inf值，跳过这一层")
                    continue
                    
                if torch.isnan(tar_feat).any() or torch.isinf(tar_feat).any():
                    print(f"警告: target_features[{i}]包含NaN或Inf值，跳过这一层")
                    continue
                
                # 限制特征值范围，防止极值
                gen_feat = torch.clamp(gen_feat, min=-100.0, max=100.0)
                tar_feat = torch.clamp(tar_feat, min=-100.0, max=100.0)
                
                # 计算当前层的损失
                layer_loss = torch.mean((tar_feat - gen_feat) ** 2)
                
                # 检查层损失是否有异常值
                if torch.isnan(layer_loss) or torch.isinf(layer_loss):
                    print(f"警告: 第{i}层损失包含NaN或Inf值，跳过")
                    continue
                
                perceptual_loss += layer_loss
                valid_layers += 1
                
            except Exception as e:
                print(f"警告: 计算第{i}层感知损失时出错: {e}，跳过这一层")
                continue
        
        # 如果没有有效层，返回零损失
        if valid_layers == 0:
            print("警告: 没有有效的VGG特征层，返回零感知损失")
            return torch.tensor(0.0, device=device)
        
        # 计算平均损失
        perceptual_loss = perceptual_loss / valid_layers
        
        # 最终数值稳定性检查
        if torch.isnan(perceptual_loss) or torch.isinf(perceptual_loss):
            print("警告: 最终感知损失包含NaN或Inf，返回零损失")
            return torch.tensor(0.0, device=device)
        
        # 限制损失范围，防止过大的值
        perceptual_loss = torch.clamp(perceptual_loss, min=0.0, max=100.0)
        
        return perceptual_loss


class EnhancedInkLoss(nn.Module):
    """
    增强的墨迹控制损失函数
    结合传统感知损失和创新的DIS损失，实现精确的墨迹扩散控制
    """
    
    def __init__(self, 
                 use_perceptual: bool = True,
                 use_dis: bool = True, 
                 dis_preset: str = 'balanced',
                 perceptual_weight: float = 1.0,
                 dis_weight: float = 0.5):
        """
        Args:
            use_perceptual: 是否使用感知损失
            use_dis: 是否使用DIS损失
            dis_preset: DIS损失预设配置
            perceptual_weight: 感知损失权重
            dis_weight: DIS损失权重
        """
        super().__init__()
        
        self.use_perceptual = use_perceptual
        self.use_dis = use_dis
        self.perceptual_weight = perceptual_weight
        self.dis_weight = dis_weight
        
        if use_perceptual:
            self.perceptual_loss = ContentPerceptualLoss()
            
        if use_dis:
            self.dis_loss = create_dis_loss_preset(dis_preset)
    
    def forward(self, generated_images, target_images, device='cuda'):
        """
        计算增强的墨迹控制损失
        
        Returns:
            包含所有损失组件的字典
        """
        losses = {}
        total_loss = 0
        
        # 1. 感知损失 - 保证整体视觉质量
        if self.use_perceptual:
            # 确保输入是3通道的RGB图像用于VGG16
            if generated_images.shape[1] == 1:
                # 灰度图转RGB：复制通道
                generated_rgb = generated_images.repeat(1, 3, 1, 1)
                target_rgb = target_images.repeat(1, 3, 1, 1)
            else:
                generated_rgb = generated_images
                target_rgb = target_images
                
            perceptual_loss = self.perceptual_loss.calculate_loss(
                generated_images=generated_rgb,
                target_images=target_rgb,
                device=device
            )
            losses['perceptual_loss'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        # 2. DIS损失 - 精确控制墨迹扩散
        if self.use_dis:
            dis_results = self.dis_loss(generated_images, target_images, return_detailed=True)
            
            # 添加DIS损失的所有组件
            for key, value in dis_results.items():
                losses[f'dis_{key}'] = value
                
            total_loss += self.dis_weight * dis_results['dis_loss']
        
        losses['total_enhanced_ink_loss'] = total_loss
        
        return losses
    
    def get_visualization_summary(self, generated_images, target_images):
        """获取可视化摘要，用于训练监控"""
        if not self.use_dis:
            return {}
            
        return self.dis_loss.get_visualization_maps(generated_images, target_images)
