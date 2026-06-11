#!/usr/bin/env python3
"""
增强的高频条件处理模块
提供更智能、更有效的高频信息条件指导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class TimeAwareHighFreqAttention(nn.Module):
    """时间步感知的高频注意力模块"""
    
    def __init__(
        self, 
        hidden_channels: int,
        high_freq_channels: int,
        time_emb_dim: int = 512,
        num_heads: int = 4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.high_freq_channels = high_freq_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        # 高频特征投影
        self.high_freq_proj = nn.Sequential(
            nn.Conv2d(high_freq_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(min(32, hidden_channels//4), hidden_channels),
            nn.SiLU()
        )
        
        # 注意力查询、键、值投影
        self.q_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.k_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.v_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # 时间步条件投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels//4), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        high_freq_feature: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        layer_index: int = 0
    ) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        
        # 1. 投影高频特征
        high_freq_aligned = self.high_freq_proj(high_freq_feature)
        
        # 2. 时间步条件调制
        if time_emb is not None:
            time_cond = self.time_proj(time_emb).view(B, C, 1, 1)
            high_freq_aligned = high_freq_aligned + time_cond
        
        # 3. 多头注意力计算
        q = self.q_proj(hidden_states).view(B, self.num_heads, self.head_dim, H*W)
        k = self.k_proj(high_freq_aligned).view(B, self.num_heads, self.head_dim, H*W)
        v = self.v_proj(high_freq_aligned).view(B, self.num_heads, self.head_dim, H*W)
        
        # 注意力分数
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attn_output = torch.matmul(v, attn_weights.transpose(-2, -1))
        attn_output = attn_output.view(B, C, H, W)
        attn_output = self.out_proj(attn_output)
        
        # 4. 自适应权重计算
        combined_features = torch.cat([hidden_states, attn_output], dim=1)
        adaptive_weight = self.weight_generator(combined_features)
        
        # 5. 层级和时间步自适应调整
        layer_factor = self._get_layer_factor(layer_index)
        time_factor = self._get_time_factor(time_emb) if time_emb is not None else 1.0
        
        final_weight = adaptive_weight * layer_factor * time_factor
        
        return hidden_states + final_weight * attn_output
    
    def _get_layer_factor(self, layer_index: int) -> float:
        """根据层级获取权重因子"""
        # 下采样阶段：浅层更依赖高频信息
        # 中间层：适中的权重
        # 上采样阶段：逐渐减少高频依赖
        if layer_index <= 1:  # 早期下采样层
            return 1.0
        elif layer_index <= 3:  # 后期下采样层
            return 0.8
        elif layer_index == 4:  # 中间层
            return 0.6
        else:  # 上采样层
            return max(0.3, 0.6 - (layer_index - 4) * 0.1)
    
    def _get_time_factor(self, time_emb: torch.Tensor) -> float:
        """根据时间步获取权重因子"""
        if time_emb is None:
            return 1.0
            
        # 将时间嵌入转换为时间步估计值
        # 早期时间步（高噪声）更依赖高频信息
        time_value = time_emb.mean().item()
        normalized_time = abs(time_value) / 1000.0  # 假设时间步在[0, 1000]
        
        # 早期时间步增强高频信息，后期时间步减少
        time_factor = 1.2 - normalized_time * 0.4
        return max(0.6, min(1.5, time_factor))


class EnhancedHighFreqCondition(nn.Module):
    """增强的高频条件处理模块"""
    
    def __init__(
        self,
        hidden_channels: int,
        high_freq_channels: int,
        time_emb_dim: int = 512,
        use_attention: bool = True,
        use_gate: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_gate = use_gate
        
        # 高频注意力模块
        if use_attention:
            self.high_freq_attention = TimeAwareHighFreqAttention(
                hidden_channels, high_freq_channels, time_emb_dim
            )
        else:
            # 简单投影作为备选
            self.high_freq_proj = nn.Sequential(
                nn.Conv2d(high_freq_channels, hidden_channels, 1, bias=False),
                nn.GroupNorm(min(32, hidden_channels//4), hidden_channels),
                nn.SiLU()
            )
        
        # 门控机制
        if use_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
                nn.GroupNorm(min(32, hidden_channels//4), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.Sigmoid()
            )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        high_freq_feature: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        layer_index: int = 0
    ) -> torch.Tensor:
        """
        增强的高频条件前向传播
        
        Args:
            hidden_states: UNet的隐藏状态 [B, C, H, W]
            high_freq_feature: 高频特征 [B, C', H', W']
            time_emb: 时间步嵌入 [B, emb_dim]
            layer_index: 层索引
        """
        # 空间对齐
        if high_freq_feature.shape[2:] != hidden_states.shape[2:]:
            high_freq_feature = F.interpolate(
                high_freq_feature,
                size=hidden_states.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if self.use_attention:
            # 使用注意力机制
            enhanced_states = self.high_freq_attention(
                hidden_states, high_freq_feature, time_emb, layer_index
            )
        else:
            # 简单投影和相加
            high_freq_aligned = self.high_freq_proj(high_freq_feature)
            enhanced_states = hidden_states + 0.3 * high_freq_aligned
        
        if self.use_gate:
            # 门控融合
            combined = torch.cat([hidden_states, enhanced_states], dim=1)
            gate = self.gate_conv(combined)
            enhanced_states = hidden_states * (1 - gate) + enhanced_states * gate
        
        # 残差连接
        output = hidden_states + self.residual_weight * (enhanced_states - hidden_states)
        
        return output


class GlobalHighFreqContextManager(nn.Module):
    """全局高频上下文管理器"""
    
    def __init__(self, num_layers: int = 8, feature_dim: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # 跨层信息传递
        self.layer_communication = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])
        
        # 全局上下文池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, layer_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        处理跨层高频特征交互
        
        Args:
            layer_features: 各层的高频特征列表
        """
        # 提取全局上下文
        global_contexts = []
        for feat in layer_features:
            global_ctx = self.global_pool(feat).squeeze(-1).squeeze(-1)
            global_contexts.append(global_ctx)
        
        # 跨层信息传递
        enhanced_contexts = []
        for i, (feat, ctx) in enumerate(zip(layer_features, global_contexts)):
            if i < len(self.layer_communication):
                enhanced_ctx = self.layer_communication[i](ctx)
                enhanced_contexts.append(enhanced_ctx)
            else:
                enhanced_contexts.append(ctx)
        
        # 将增强的上下文重新注入特征
        enhanced_features = []
        for feat, enhanced_ctx in zip(layer_features, enhanced_contexts):
            B, C, H, W = feat.shape
            ctx_broadcast = enhanced_ctx.view(B, C, 1, 1).expand_as(feat)
            enhanced_feat = feat + 0.1 * ctx_broadcast
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


def build_enhanced_high_freq_condition(
    hidden_channels: int,
    high_freq_channels: int,
    time_emb_dim: int = 512,
    condition_type: str = "attention"
) -> nn.Module:
    """构建增强的高频条件处理模块"""
    
    if condition_type == "attention":
        return EnhancedHighFreqCondition(
            hidden_channels=hidden_channels,
            high_freq_channels=high_freq_channels,
            time_emb_dim=time_emb_dim,
            use_attention=True,
            use_gate=True
        )
    elif condition_type == "gate":
        return EnhancedHighFreqCondition(
            hidden_channels=hidden_channels,
            high_freq_channels=high_freq_channels,
            time_emb_dim=time_emb_dim,
            use_attention=False,
            use_gate=True
        )
    else:  # simple
        return EnhancedHighFreqCondition(
            hidden_channels=hidden_channels,
            high_freq_channels=high_freq_channels,
            time_emb_dim=time_emb_dim,
            use_attention=False,
            use_gate=False
        )


# 使用示例和测试代码
if __name__ == "__main__":
    # 测试增强的高频条件模块
    B, C, H, W = 2, 256, 24, 24
    hidden_states = torch.randn(B, C, H, W)
    high_freq_feature = torch.randn(B, 256, H, W)
    time_emb = torch.randn(B, 512)
    
    # 创建增强模块
    enhanced_condition = build_enhanced_high_freq_condition(
        hidden_channels=C,
        high_freq_channels=256,
        condition_type="attention"
    )
    
    # 前向传播
    output = enhanced_condition(
        hidden_states, high_freq_feature, time_emb, layer_index=1
    )
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 增强的高频条件模块测试通过!") 