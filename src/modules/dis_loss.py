import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DifferentiableInkStructureLoss(nn.Module):

    
    def __init__(self, 
                 lambda_core: float = 1.0,
                 lambda_boundary: float = 0.8, 
                 lambda_edge: float = 0.5,       # 对应论文中的 lambda_edge
                 beta: float = 0.5,               # 论文公式(10)的内外带平衡因子
                 eta: float = 0.5,                # 论文公式(11)平衡梯度和拉普拉斯的权重
                 erosion_kernel_size: int = 3,
                 dilation_kernel_size: int = 5,
                 adaptive_weights: bool = True,
                 temperature: float = 0.1):       # 降低温度使Soft操作更逼近真实形态学
        super().__init__()
        
        self.lambda_core = lambda_core
        self.lambda_boundary = lambda_boundary  
        self.lambda_edge = lambda_edge
        self.beta = beta
        self.eta = eta
        self.erosion_kernel_size = erosion_kernel_size
        self.dilation_kernel_size = dilation_kernel_size
        self.adaptive_weights = adaptive_weights
        self.temperature = temperature
        
        # 注册形态学核
        self._register_morphology_kernels()
        
        # 注册一阶 Sobel 算子（用于计算公式11的 ∇I）
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # 注册二阶 Laplacian 算子（用于计算公式11的 ΔI）
        self.register_buffer('laplacian_kernel', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

        if adaptive_weights:
            self.register_buffer('loss_history', torch.zeros(3, 100))
            self.register_buffer('history_idx', torch.tensor(0))
    
    def _register_morphology_kernels(self):
        erosion_kernel = self._create_circular_kernel(self.erosion_kernel_size)
        self.register_buffer('erosion_kernel', erosion_kernel)

        dilation_kernel = self._create_circular_kernel(self.dilation_kernel_size) 
        self.register_buffer('dilation_kernel', dilation_kernel)
    
    def _create_circular_kernel(self, size: int) -> torch.Tensor:
        kernel = torch.zeros(size, size)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                    kernel[i, j] = 1.0
        # 归一化核，使其在卷积时自动计算局部均值，完美契合 Log-Sum-Exp 的 1/|Ω|
        return kernel.view(1, 1, size, size) / kernel.sum()
    
    def to_ink_map(self, img: torch.Tensor) -> torch.Tensor:
        """转换为灰度图并取反，转换为墨迹响应图 I = 1 - x (墨迹处为1，背景处为0)"""
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        return 1.0 - gray
    
    def soft_erosion(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        完全对齐论文公式(6)的数值稳定版可微分软腐蚀
        Formula: -tau * log( 1/|Ω| * sum( exp(-I / tau) ) )
        """
        padding = kernel.shape[-1] // 2
        # 数值稳定技巧：减去最大值防止 exp 爆炸
        scaled = -img / (self.temperature + 1e-8)
        max_val = torch.max(scaled)
        exp_term = torch.exp(scaled - max_val)
        
        conv_result = F.conv2d(exp_term, kernel, padding=padding)
        soft_min = -self.temperature * (torch.log(conv_result + 1e-8) + max_val)
        return soft_min
    
    def soft_dilation(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        完全对齐论文公式(7)的数值稳定版可微分软膨胀
        Formula: tau * log( 1/|Ω| * sum( exp(I / tau) ) )
        """
        padding = kernel.shape[-1] // 2
        # 数值稳定技巧
        scaled = img / (self.temperature + 1e-8)
        max_val = torch.max(scaled)
        exp_term = torch.exp(scaled - max_val)
        
        conv_result = F.conv2d(exp_term, kernel, padding=padding)
        soft_max = self.temperature * (torch.log(conv_result + 1e-8) + max_val)
        return soft_max

    def _update_adaptive_weights(self, core_loss: torch.Tensor, boundary_loss: torch.Tensor, edge_loss: torch.Tensor):
        if not self.adaptive_weights:
            return
            
        current_losses = torch.stack([core_loss.detach(), boundary_loss.detach(), edge_loss.detach()])
        idx = self.history_idx.item() % 100
        self.loss_history[:, idx] = current_losses
        self.history_idx += 1

        if self.history_idx >= 20:
            recent_window = min(20, self.history_idx.item())
            recent_losses = self.loss_history[:, max(0, idx-recent_window+1):idx+1]
            loss_stds = torch.std(recent_losses, dim=1)
            total_std = torch.sum(loss_stds) + 1e-8
            adaptive_factors = loss_stds / total_std

            momentum = 0.9
            self.lambda_core = momentum * self.lambda_core + (1 - momentum) * (1.0 + adaptive_factors[0])
            self.lambda_boundary = momentum * self.lambda_boundary + (1 - momentum) * (0.8 + adaptive_factors[1])  
            self.lambda_edge = momentum * self.lambda_edge + (1 - momentum) * (0.5 + adaptive_factors[2])

    def forward(self, generated_img: torch.Tensor, target_img: torch.Tensor, 
                return_detailed: bool = True) -> Dict[str, torch.Tensor]:
        
        # 1. 转换为墨迹响应图 I = 1 - x
        I_g = self.to_ink_map(generated_img)
        I_t = self.to_ink_map(target_img)
        
        # 尺度对齐保护
        if I_g.shape != I_t.shape:
            min_h = min(I_g.shape[2], I_t.shape[2])
            min_w = min(I_g.shape[3], I_t.shape[3])
            I_g = I_g[:, :, :min_h, :min_w]
            I_t = I_t[:, :, :min_h, :min_w]

        # 2. 一键计算基础形态学组件（避免子函数重复卷积，提速3倍）
        I_g_eroded = self.soft_erosion(I_g, self.erosion_kernel)
        I_t_eroded = self.soft_erosion(I_t, self.erosion_kernel)
        I_t_dilated = self.soft_dilation(I_t, self.dilation_kernel)

        # 3. 计算 Stroke-Core 骨架一致性损失 (公式 12)
        core_loss = F.l1_loss(I_g_eroded, I_t_eroded)

        # 4. 精准构造完全对齐论文的扩散带掩膜 M_band (公式 8, 9, 10)
        M_inner = torch.clamp(I_t - I_t_eroded, min=0.0)
        M_outer = torch.clamp(I_t_dilated - I_t, min=0.0)
        M_band = torch.clamp(M_inner + self.beta * M_outer, min=0.0, max=1.0)

        # 5. 计算 Diffusion-Band 边界一致性损失 (公式 13)
        boundary_loss = torch.mean(torch.abs(I_g - I_t) * M_band)

        # 6. 计算 Edge-Response 梯度及拉普拉斯损失 (公式 11)
        # 一阶梯度 ∇I
        gen_grad_x = F.conv2d(I_g, self.sobel_x, padding=1)
        gen_grad_y = F.conv2d(I_g, self.sobel_y, padding=1)
        I_g_grad = torch.sqrt(gen_grad_x**2 + gen_grad_y**2 + 1e-6)

        tar_grad_x = F.conv2d(I_t, self.sobel_x, padding=1)
        tar_grad_y = F.conv2d(I_t, self.sobel_y, padding=1)
        I_t_grad = torch.sqrt(tar_grad_x**2 + tar_grad_y**2 + 1e-6)
        
        grad_loss = torch.mean(torch.abs(I_g_grad - I_t_grad) * M_band)

        # 二阶拉普拉斯 ΔI
        I_g_lap = F.conv2d(I_g, self.laplacian_kernel, padding=1)
        I_t_lap = F.conv2d(I_t, self.laplacian_kernel, padding=1)
        
        lap_loss = torch.mean(torch.abs(I_g_lap - I_t_lap) * M_band)

        # 联合形成最终的 edge 损失
        edge_loss = grad_loss + self.eta * lap_loss

        # 7. 更新自适应权重
        self._update_adaptive_weights(core_loss, boundary_loss, edge_loss)

        # 8. 联合总 DIS Loss (公式 14)
        total_dis_loss = (self.lambda_core * core_loss + 
                          self.lambda_boundary * boundary_loss + 
                          self.lambda_edge * edge_loss)
        
        results = {
            'dis_loss': total_dis_loss,
            'core_loss': core_loss,
            'boundary_loss': boundary_loss, 
            'edge_loss': edge_loss
        }
        
        if return_detailed:
            results.update({
                'lambda_core': torch.tensor(self.lambda_core),
                'lambda_boundary': torch.tensor(self.lambda_boundary),
                'lambda_edge': torch.tensor(self.lambda_edge),
                'laplacian_sub_loss': lap_loss,
                'gradient_sub_loss': grad_loss
            })
        
        return results