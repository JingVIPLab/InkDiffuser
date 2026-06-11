#!/usr/bin/env python3
"""
增强的条件指导模块 - 解决条件指导模式效果不佳的问题
主要改进：
1. 可学习的层级权重
2. 时间步感知的权重调节
3. 注意力机制的引入
4. 全局信息协调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LearnableConditionWeights(nn.Module):
    """可学习的条件权重模块"""
    
    def __init__(self, num_layers: int = 8, time_emb_dim: int = 512):
        super().__init__()
        # 为每个UNet层级创建可学习权重
        self.layer_weights = nn.Parameter(
            torch.tensor([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15][:num_layers])
        )
        
        # 时间步调制网络
        self.time_modulation = nn.Sequential(
            nn.Linear(time_emb_dim, num_layers),
            nn.Tanh()  # 输出[-1, 1]，用于调制权重
        )
        
        # 层级特定的调制
        self.layer_modulation = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, layer_index: int, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 基础权重
        base_weight = torch.sigmoid(self.layer_weights[layer_index])
        
        # 时间步调制
        if time_emb is not None:
            time_mod = self.time_modulation(time_emb.mean(dim=0))  # [num_layers]
            time_factor = 1.0 + 0.3 * time_mod[layer_index]  # 调制范围[0.7, 1.3]
        else:
            time_factor = 1.0
        
        # 层级调制
        layer_factor = torch.sigmoid(self.layer_modulation[layer_index])
        
        final_weight = base_weight * time_factor * layer_factor
        return torch.clamp(final_weight, 0.05, 0.8)  # 约束在合理范围


class ConditionAttention(nn.Module):
    """条件注意力模块"""
    
    def __init__(self, hidden_channels: int, condition_channels: int, num_heads: int = 4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 投影层
        self.q_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.k_proj = nn.Conv2d(condition_channels, hidden_channels, 1)
        self.v_proj = nn.Conv2d(condition_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # 归一化
        self.norm_q = nn.GroupNorm(min(32, hidden_channels//8), hidden_channels)
        self.norm_k = nn.GroupNorm(min(32, hidden_channels//8), hidden_channels)
        
    def forward(self, hidden_states: torch.Tensor, condition_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        
        # 投影
        q = self.norm_q(self.q_proj(hidden_states))
        k = self.norm_k(self.k_proj(condition_feat))
        v = self.v_proj(condition_feat)
        
        # 重塑为多头
        q = q.view(B, self.num_heads, self.head_dim, H*W)
        k = k.view(B, self.num_heads, self.head_dim, H*W)
        v = v.view(B, self.num_heads, self.head_dim, H*W)
        
        # 注意力计算
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attn_out = torch.matmul(v, attn_weights.transpose(-2, -1))
        attn_out = attn_out.view(B, C, H, W)
        
        return self.out_proj(attn_out)


class AdaptiveConditionFusion(nn.Module):
    """自适应条件融合模块"""
    
    def __init__(self, hidden_channels: int, condition_channels: int):
        super().__init__()
        
        # 通道对齐
        if condition_channels != hidden_channels:
            self.channel_align = nn.Sequential(
                nn.Conv2d(condition_channels, hidden_channels, 1, bias=False),
                nn.GroupNorm(min(32, hidden_channels//8), hidden_channels),
                nn.SiLU()
            )
        else:
            self.channel_align = nn.Identity()
        
        # 注意力模块
        self.attention = ConditionAttention(hidden_channels, hidden_channels)
        
        # 门控网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels//8), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # 特征精炼
        self.feature_refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels//8), hidden_channels),
            nn.SiLU()
        )
        
    def forward(self, hidden_states: torch.Tensor, condition_feat: torch.Tensor) -> torch.Tensor:
        # 1. 通道对齐
        aligned_condition = self.channel_align(condition_feat)
        
        # 2. 空间对齐
        if aligned_condition.shape[2:] != hidden_states.shape[2:]:
            aligned_condition = F.interpolate(
                aligned_condition, 
                size=hidden_states.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # 3. 注意力融合
        attended = self.attention(hidden_states, aligned_condition)
        
        # 4. 门控融合
        combined = torch.cat([hidden_states, attended], dim=1)
        gate = self.gate_conv(combined)
        gated = hidden_states * (1 - gate) + attended * gate
        
        # 5. 特征精炼
        refined = self.feature_refine(gated)
        
        return refined


class EnhancedConditionGuidance(nn.Module):
    """增强的条件指导模块"""
    
    def __init__(
        self, 
        num_layers: int = 8,
        time_emb_dim: int = 512,
        use_attention: bool = True,
        use_global_coordination: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_global_coordination = use_global_coordination
        
        # 可学习权重
        self.weight_controller = LearnableConditionWeights(num_layers, time_emb_dim)
        
        # 条件融合模块字典（延迟初始化）
        self.condition_fusions = nn.ModuleDict()
        
        # 全局协调器
        if use_global_coordination:
            self.global_coordinator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(256, 64, 1),    # 降维
                nn.SiLU(),
                nn.Conv2d(64, num_layers, 1),  # 输出每层权重
                nn.Sigmoid()
            )
        
    def get_or_create_fusion(self, hidden_channels: int, condition_channels: int, layer_idx: int):
        """获取或创建融合模块"""
        key = f"layer_{layer_idx}_{hidden_channels}_{condition_channels}"
        
        if key not in self.condition_fusions:
            if self.use_attention:
                self.condition_fusions[key] = AdaptiveConditionFusion(
                    hidden_channels, condition_channels
                )
            else:
                # 简化版本
                if condition_channels != hidden_channels:
                    self.condition_fusions[key] = nn.Sequential(
                        nn.Conv2d(condition_channels, hidden_channels, 1, bias=False),
                        nn.GroupNorm(min(32, hidden_channels//8), hidden_channels),
                        nn.SiLU()
                    )
                else:
                    self.condition_fusions[key] = nn.Identity()
        
        return self.condition_fusions[key]
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        condition_feature: torch.Tensor,
        layer_index: int,
        time_emb: Optional[torch.Tensor] = None,
        global_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        增强的条件指导前向传播
        
        Args:
            hidden_states: UNet层的隐藏状态
            condition_feature: 对应的条件特征
            layer_index: 当前层索引
            time_emb: 时间步嵌入
            global_condition: 全局条件信息
        """
        
        # 1. 获取自适应权重
        adaptive_weight = self.weight_controller(layer_index, time_emb)
        
        # 2. 全局协调（如果启用）
        if self.use_global_coordination and global_condition is not None:
            global_weights = self.global_coordinator(global_condition)  # [B, num_layers, 1, 1]
            global_weight = global_weights[:, layer_index:layer_index+1, :, :]  # [B, 1, 1, 1]
            adaptive_weight = adaptive_weight * global_weight.squeeze()
        
        # 3. 获取融合模块
        fusion_module = self.get_or_create_fusion(
            hidden_states.shape[1], 
            condition_feature.shape[1], 
            layer_index
        )
        
        # 4. 条件融合
        if self.use_attention:
            enhanced_states = fusion_module(hidden_states, condition_feature)
        else:
            # 简化融合
            aligned_condition = fusion_module(condition_feature)
            if aligned_condition.shape[2:] != hidden_states.shape[2:]:
                aligned_condition = F.interpolate(
                    aligned_condition,
                    size=hidden_states.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            enhanced_states = hidden_states + aligned_condition
        
        # 5. 权重应用
        output = hidden_states + adaptive_weight * (enhanced_states - hidden_states)
        
        return output


# 使用示例
def create_enhanced_condition_guidance(
    num_layers: int = 8,
    condition_type: str = "enhanced"  # "enhanced", "attention", "simple"
) -> EnhancedConditionGuidance:
    """创建增强的条件指导模块"""
    
    if condition_type == "enhanced":
        return EnhancedConditionGuidance(
            num_layers=num_layers,
            use_attention=True,
            use_global_coordination=True
        )
    elif condition_type == "attention":
        return EnhancedConditionGuidance(
            num_layers=num_layers,
            use_attention=True,
            use_global_coordination=False
        )
    else:  # simple
        return EnhancedConditionGuidance(
            num_layers=num_layers,
            use_attention=False,
            use_global_coordination=False
        )


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    B, C, H, W = 2, 256, 24, 24
    hidden_states = torch.randn(B, C, H, W).to(device)
    condition_feature = torch.randn(B, 256, H, W).to(device)
    time_emb = torch.randn(B, 512).to(device)
    global_condition = torch.randn(B, 256, H, W).to(device)
    
    # 创建增强模块
    enhanced_guidance = create_enhanced_condition_guidance(
        num_layers=8, condition_type="enhanced"
    ).to(device)
    
    # 测试前向传播
    output = enhanced_guidance(
        hidden_states, condition_feature, 
        layer_index=2, time_emb=time_emb, 
        global_condition=global_condition
    )
    
    print(f"✅ 增强条件指导测试通过！")
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    print(f"权重范围: {enhanced_guidance.weight_controller.layer_weights.data}") 