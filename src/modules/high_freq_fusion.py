import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class ChannelAttention(nn.Module):
    """通道注意力模块 - SE-Block风格"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 合并并应用sigmoid
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return out


class SpatialAttention(nn.Module):
    """空间注意力模块 - CBAM风格"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道维度统计
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return attention


class CrossModalAttention(nn.Module):
    """交叉模态注意力模块"""
    
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        
        self.q_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        
        self.norm_q = nn.LayerNorm(channels)
        self.norm_k = nn.LayerNorm(channels)
        self.norm_v = nn.LayerNorm(channels)
        
    def forward(self, content_feat: torch.Tensor, high_freq_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = content_feat.shape
        
        # 生成Q, K, V
        q = self.q_conv(content_feat).view(B, self.heads, self.head_dim, H*W)
        k = self.k_conv(high_freq_feat).view(B, self.heads, self.head_dim, H*W)
        v = self.v_conv(high_freq_feat).view(B, self.heads, self.head_dim, H*W)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attended = torch.matmul(v, attn_weights.transpose(-2, -1))
        attended = attended.view(B, C, H, W)
        
        # 输出投影
        output = self.out_conv(attended)
        
        return output


class AdaptiveGateNetwork(nn.Module):
    """自适应门控网络"""
    
    def __init__(self, content_channels: int, high_freq_channels: int, time_emb_dim: int = 512):
        super().__init__()
        
        # 特征对齐
        self.feat_align = nn.Conv2d(content_channels + high_freq_channels, content_channels, 1)
        
        # 时间嵌入处理
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, content_channels),
            nn.SiLU(),
            nn.Linear(content_channels, content_channels)
        )
        
        # 门控网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(content_channels * 2, content_channels, 3, padding=1),
            nn.BatchNorm2d(content_channels),
            nn.SiLU(),
            nn.Conv2d(content_channels, content_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, content_feat, high_freq_feat, time_emb=None):
        B, C, H, W = content_feat.shape
        
        # 特征拼接和对齐
        combined = torch.cat([content_feat, high_freq_feat], dim=1)
        aligned = self.feat_align(combined)
        
        # 时间条件
        if time_emb is not None:
            time_cond = self.time_mlp(time_emb).view(B, C, 1, 1)
            aligned = aligned + time_cond
        
        # 门控计算
        gate_input = torch.cat([content_feat, aligned], dim=1)
        gate = self.gate_conv(gate_input)
        
        return gate


class AdaptiveHighFreqFusion(nn.Module):
    """自适应高频特征融合模块"""
    
    def __init__(
        self,
        content_channels: int,
        high_freq_channels: int,
        reduction: int = 16,
        fusion_type: str = "adaptive",
        time_emb_dim: int = 512
    ):
        super().__init__()
        self.content_channels = content_channels
        self.high_freq_channels = high_freq_channels
        self.fusion_type = fusion_type
        
        # 通道对齐
        if high_freq_channels != content_channels:
            self.channel_align = nn.Sequential(
                nn.Conv2d(high_freq_channels, content_channels, 1, bias=False),
                nn.BatchNorm2d(content_channels),
                nn.SiLU()
            )
        else:
            self.channel_align = nn.Identity()
        
        # 注意力模块
        self.channel_attention = ChannelAttention(content_channels, reduction)
        self.spatial_attention = SpatialAttention()
        self.cross_attention = CrossModalAttention(content_channels)
        
        # 门控网络
        self.gate_network = AdaptiveGateNetwork(content_channels, content_channels, time_emb_dim)
        
        # 特征精炼
        self.feature_refine = nn.Sequential(
            nn.Conv2d(content_channels, content_channels, 3, padding=1),
            nn.BatchNorm2d(content_channels),
            nn.SiLU(),
            nn.Conv2d(content_channels, content_channels, 1)
        )
        
        # 自适应权重（可学习参数）
        self.adaptive_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # [gate, cross_attn, channel_attn]
        
    def _align_features(self, high_freq_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
        """特征对齐：尺寸和通道"""
        # 通道对齐
        high_freq_aligned = self.channel_align(high_freq_feat)
        
        # 空间尺寸对齐
        if high_freq_aligned.shape[2:] != content_feat.shape[2:]:
            high_freq_aligned = F.interpolate(
                high_freq_aligned, 
                size=content_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return high_freq_aligned
    
    def forward(
        self, 
        content_feat: torch.Tensor, 
        high_freq_feat: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
        layer_index: int = 0
    ) -> torch.Tensor:
        """
        智能融合前向传播
        
        Args:
            content_feat: 内容特征 [B, C, H, W]
            high_freq_feat: 高频特征 [B, C', H', W']
            timestep_emb: 时间步嵌入 [B, emb_dim]
            layer_index: 层索引
        """
        # 1. 特征对齐
        high_freq_aligned = self._align_features(high_freq_feat, content_feat)
        
        # 2. 根据融合类型选择策略
        if self.fusion_type == "simple":
            return self._simple_fusion(content_feat, high_freq_aligned)
        elif self.fusion_type == "attention":
            return self._attention_fusion(content_feat, high_freq_aligned)
        elif self.fusion_type == "gate":
            return self._gate_fusion(content_feat, high_freq_aligned, timestep_emb)
        else:  # adaptive
            return self._adaptive_fusion(content_feat, high_freq_aligned, timestep_emb, layer_index)
    
    def _simple_fusion(self, content_feat: torch.Tensor, high_freq_feat: torch.Tensor) -> torch.Tensor:
        """改进的简单融合"""
        # 通道注意力加权
        channel_weight = self.channel_attention(high_freq_feat)
        weighted_high_freq = high_freq_feat * channel_weight
        
        return content_feat + 0.3 * weighted_high_freq
    
    def _attention_fusion(self, content_feat: torch.Tensor, high_freq_feat: torch.Tensor) -> torch.Tensor:
        """基于注意力的融合"""
        # 交叉模态注意力
        cross_attended = self.cross_attention(content_feat, high_freq_feat)
        
        # 空间注意力
        spatial_weight = self.spatial_attention(cross_attended)
        spatial_attended = cross_attended * spatial_weight
        
        # 通道注意力
        channel_weight = self.channel_attention(spatial_attended)
        final_attended = spatial_attended * channel_weight
        
        return content_feat + final_attended
    
    def _gate_fusion(self, content_feat: torch.Tensor, high_freq_feat: torch.Tensor, timestep_emb: Optional[torch.Tensor]) -> torch.Tensor:
        """门控融合"""
        # 计算自适应门控
        gate = self.gate_network(content_feat, high_freq_feat, timestep_emb)
        
        # 门控融合
        fused = content_feat * (1 - gate) + high_freq_feat * gate
        
        return fused
    
    def _adaptive_fusion(
        self, 
        content_feat: torch.Tensor, 
        high_freq_feat: torch.Tensor,
        timestep_emb: Optional[torch.Tensor],
        layer_index: int
    ) -> torch.Tensor:
        """自适应多策略融合"""
        
        # 1. 门控融合分支
        gate_result = self._gate_fusion(content_feat, high_freq_feat, timestep_emb)
        
        # 2. 交叉注意力分支
        cross_attn_result = self.cross_attention(content_feat, high_freq_feat)
        
        # 3. 通道注意力分支
        channel_weight = self.channel_attention(high_freq_feat)
        channel_attn_result = high_freq_feat * channel_weight
        
        # 4. 计算自适应权重
        weights = self._compute_adaptive_weights(timestep_emb, layer_index)
        
        # 5. 加权组合
        fused = (weights[0] * gate_result + 
                weights[1] * cross_attn_result + 
                weights[2] * channel_attn_result)
        
        # 6. 特征精炼
        refined = self.feature_refine(fused)
        
        # 7. 残差连接
        return content_feat + refined
    
    def _compute_adaptive_weights(
        self, 
        timestep_emb: Optional[torch.Tensor], 
        layer_index: int
    ) -> torch.Tensor:
        """计算自适应权重"""
        weights = self.adaptive_weights.clone()
        
        # 根据层深度调整权重
        layer_factor = min(layer_index / 4.0, 1.0)  # 限制在[0,1]
        
        # 浅层更依赖门控，深层更依赖注意力
        weights[0] *= (1.0 - layer_factor * 0.5)  # 门控权重
        weights[1] *= (0.5 + layer_factor * 0.5)  # 交叉注意力权重
        weights[2] *= (0.8 + layer_factor * 0.2)  # 通道注意力权重
        
        # 如果有时间步信息，进一步调整
        if timestep_emb is not None:
            # 处理时间步信息 - 确保是张量格式并转换为浮点数
            if torch.is_tensor(timestep_emb):
                # 转换为浮点数类型避免mean()错误
                timestep_float = timestep_emb.float()
                if timestep_float.numel() > 1:
                    time_value = timestep_float.mean().item()
                else:
                    time_value = timestep_float.item()
            else:
                time_value = float(timestep_emb)
            
            # 假设时间步范围在[0, 1000]，早期时间步更依赖高频信息
            time_factor = time_value / 1000.0
            time_factor = max(0.0, min(1.0, time_factor))
            
            # 早期时间步增强注意力机制
            weights[1] *= (1.0 + (1.0 - time_factor) * 0.3)
            weights[2] *= (1.0 + (1.0 - time_factor) * 0.2)
        
        # 归一化权重
        weights = F.softmax(weights, dim=0)
        
        return weights


class LayerSpecificFusion(nn.Module):
    """层级特定的融合策略"""
    
    def __init__(self, num_layers: int, base_channels: int, time_emb_dim: int = 512):
        super().__init__()
        self.num_layers = num_layers
        
        # 为每一层创建特定的融合模块
        self.layer_fusions = nn.ModuleList()
        
        for i in range(num_layers):
            # 根据层级选择不同的融合策略
            if i <= 1:  # 浅层：更多依赖空间细节
                fusion_type = "attention"
            elif i <= 3:  # 中层：平衡门控和注意力
                fusion_type = "gate"
            else:  # 深层：更复杂的自适应融合
                fusion_type = "adaptive"
            
            channels = base_channels * (2 ** min(i, 4))  # 限制通道数增长
            
            fusion_module = AdaptiveHighFreqFusion(
                content_channels=channels,
                high_freq_channels=channels,
                fusion_type=fusion_type,
                time_emb_dim=time_emb_dim
            )
            
            self.layer_fusions.append(fusion_module)
    
    def forward(
        self, 
        content_features: List[torch.Tensor], 
        high_freq_features: List[torch.Tensor],
        timestep_emb: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        对多层特征进行层级特定的融合
        """
        fused_features = []
        
        for i, (content_feat, high_freq_feat) in enumerate(zip(content_features, high_freq_features)):
            if i < len(self.layer_fusions):
                fused = self.layer_fusions[i](
                    content_feat, high_freq_feat, timestep_emb, i
                )
                fused_features.append(fused)
            else:
                # 超出预定义范围的层，使用简单融合
                if content_feat.shape == high_freq_feat.shape:
                    fused_features.append(content_feat + 0.2 * high_freq_feat)
                else:
                    fused_features.append(content_feat)
        
        return fused_features


def build_fusion_module(content_channels: int, high_freq_channels: int, fusion_type: str = "adaptive"):
    """构建融合模块的工厂函数 - 简化版"""
    return SimpleStableFusion(
        content_channels=content_channels,
        high_freq_channels=high_freq_channels,
        fusion_type=fusion_type
    )


class SimpleStableFusion(nn.Module):
    """简单稳定的高频特征融合模块"""
    
    def __init__(
        self,
        content_channels: int,
        high_freq_channels: int,
        fusion_type: str = "adaptive"
    ):
        super().__init__()
        self.content_channels = content_channels
        self.high_freq_channels = high_freq_channels
        self.fusion_type = fusion_type
        
        # 通道对齐 - 简单的1x1卷积
        if high_freq_channels != content_channels:
            self.channel_align = nn.Conv2d(high_freq_channels, content_channels, 1, bias=False)
            nn.init.xavier_uniform_(self.channel_align.weight)
        else:
            self.channel_align = nn.Identity()
        
        # 简单的注意力权重生成
        self.attention_conv = nn.Sequential(
            nn.Conv2d(content_channels, max(1, content_channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, content_channels // 4), 1, 1),
            nn.Sigmoid()
        )
        
        # 融合权重 - 可学习的标量
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(
        self, 
        content_feat: torch.Tensor, 
        high_freq_feat: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
        layer_index: int = 0
    ) -> torch.Tensor:
        """
        简单稳定的融合前向传播
        """
        # 1. 通道对齐
        high_freq_aligned = self.channel_align(high_freq_feat)
        
        # 2. 空间对齐
        if high_freq_aligned.shape[2:] != content_feat.shape[2:]:
            high_freq_aligned = F.interpolate(
                high_freq_aligned, 
                size=content_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 3. 生成注意力权重
        attention_weight = self.attention_conv(high_freq_aligned)
        
        # 4. 应用注意力加权
        weighted_high_freq = high_freq_aligned * attention_weight
        
        # 5. 自适应权重调整（基于层级和时间步）
        adaptive_weight = self.fusion_weight
        
        # 根据层级调整权重（浅层更依赖高频信息）
        layer_factor = max(0.1, 1.0 - layer_index * 0.15)
        adaptive_weight = adaptive_weight * layer_factor
        
        # 如果有时间步信息，早期时间步增强高频信息
        if timestep_emb is not None:
            # 处理时间步信息 - 确保是张量格式并转换为浮点数
            if torch.is_tensor(timestep_emb):
                # 转换为浮点数类型避免mean()错误
                timestep_float = timestep_emb.float()
                if timestep_float.numel() > 1:
                    time_value = timestep_float.mean().item()
                else:
                    time_value = timestep_float.item()
            else:
                time_value = float(timestep_emb)
            
            # 假设时间步范围在[0, 1000]，早期时间步更依赖高频信息
            time_factor = 1.0 + (1000.0 - time_value) / 1000.0 * 0.2
            time_factor = max(0.8, min(1.5, time_factor))  # 限制范围
            adaptive_weight = adaptive_weight * time_factor
        
        # 限制权重范围，避免过大的影响
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1.0)
        
        # 6. 融合输出
        fused = content_feat + adaptive_weight * weighted_high_freq
        
        return fused 