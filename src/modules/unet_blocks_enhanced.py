#!/usr/bin/env python3
"""
增强版UNet块 - 集成改进的条件指导机制
解决条件指导模式效果不佳的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhanced_condition_guidance import create_enhanced_condition_guidance


class EnhancedMCADownBlock2D(nn.Module):
    """增强的MCA下采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        channel_attn: bool = False,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        content_channel=16,
        reduction=32,
        # 新增参数
        enhanced_condition_guidance: bool = True,
        condition_guidance_type: str = "enhanced",  # "enhanced", "attention", "simple"
    ):
        super().__init__()
        
        # 导入必要的模块
        from .attention import ChannelAttnBlock, SpatialTransformer
        from .resnet import ResnetBlock2D
        from .upsample import Downsample2D
        
        content_attentions = []
        resnets = []
        style_attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.enhanced_condition_guidance = enhanced_condition_guidance

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels+content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            style_attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
        
        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    out_channels, use_conv=True, out_channels=out_channels, 
                    padding=downsample_padding, name="op"
                )
            ])
        else:
            self.downsamplers = None

        # 🚀 新增：增强的条件指导模块
        if enhanced_condition_guidance:
            self.condition_guidance = create_enhanced_condition_guidance(
                num_layers=8,  # UNet总层数
                condition_type=condition_guidance_type
            )
            print(f"✅ 创建{condition_guidance_type}类型的增强条件指导模块")
        
        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states, 
        index,
        temb=None, 
        encoder_hidden_states=None
    ):
        output_states = ()

        for layer_idx, (content_attn, resnet, style_attn) in enumerate(
            zip(self.content_attentions, self.resnets, self.style_attentions)
        ):
            
            # 原有的内容注意力
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # 🚀 增强的高频条件处理
            if (self.enhanced_condition_guidance and 
                len(encoder_hidden_states) > 4 and 
                encoder_hidden_states[4] is not None):
                
                try:
                    high_freq_features = encoder_hidden_states[4]
                    if index < len(high_freq_features):
                        current_high_freq_feature = high_freq_features[index]
                        
                        # 使用增强的条件指导
                        hidden_states = self.condition_guidance(
                            hidden_states=hidden_states,
                            condition_feature=current_high_freq_feature,
                            layer_index=index,  # 使用全局层索引
                            time_emb=temb,
                            global_condition=high_freq_features[-1] if len(high_freq_features) > 1 else None
                        )
                        
                except Exception as e:
                    print(f"Warning: 增强条件指导失败 (layer {index}): {e}")
                    # 降级到简单处理
                    self._fallback_condition_processing(
                        hidden_states, encoder_hidden_states, index
                    )
            
            # 时间嵌入
            hidden_states = resnet(hidden_states, temb)

            # 风格注意力
            current_style_feature = encoder_hidden_states[0]
            if len(current_style_feature.shape) == 4:
                batch_size, channel, height, width = current_style_feature.shape
                current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(
                    batch_size, height*width, channel
                )
            hidden_states = style_attn(hidden_states, context=current_style_feature)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states
    
    def _fallback_condition_processing(self, hidden_states, encoder_hidden_states, index):
        """降级的条件处理"""
        try:
            high_freq_features = encoder_hidden_states[4]
            if index < len(high_freq_features):
                current_high_freq_feature = high_freq_features[index]
                
                # 简单的相加处理
                if current_high_freq_feature.shape[2:] != hidden_states.shape[2:]:
                    current_high_freq_feature = F.interpolate(
                        current_high_freq_feature, 
                        size=hidden_states.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 简单的权重和投影
                if current_high_freq_feature.shape[1] != hidden_states.shape[1]:
                    # 简单投影
                    proj = nn.Conv2d(
                        current_high_freq_feature.shape[1], 
                        hidden_states.shape[1], 
                        1, bias=False
                    ).to(hidden_states.device)
                    current_high_freq_feature = proj(current_high_freq_feature)
                
                # 固定权重相加
                hidden_states = hidden_states + 0.3 * current_high_freq_feature
                
        except Exception as e:
            print(f"Warning: 降级条件处理也失败: {e}")


class EnhancedUNetMidMCABlock2D(nn.Module):
    """增强的UNet中间MCA块"""
    
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        channel_attn: bool = False,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        content_channel=256,
        reduction=32,
        # 新增参数
        enhanced_condition_guidance: bool = True,
        condition_guidance_type: str = "enhanced",
        **kwargs,
    ):
        super().__init__()
        
        from .attention import ChannelAttnBlock, SpatialTransformer
        from .resnet import ResnetBlock2D
        
        self.enhanced_condition_guidance = enhanced_condition_guidance
        
        # 原有的层结构
        content_attentions = []
        resnets = []
        style_attentions = []

        for i in range(num_layers):
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels+content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            style_attentions.append(
                SpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )

        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)
        
        # 🚀 增强的条件指导模块
        if enhanced_condition_guidance:
            self.condition_guidance = create_enhanced_condition_guidance(
                num_layers=8,
                condition_type=condition_guidance_type
            )

    def forward(
        self, 
        hidden_states, 
        temb=None, 
        encoder_hidden_states=None,
        index=None,
    ):
        for content_attn, resnet, style_attn in zip(
            self.content_attentions, self.resnets, self.style_attentions
        ):
            
            # 内容注意力
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # 🚀 增强的条件指导（中间层）
            if (self.enhanced_condition_guidance and 
                len(encoder_hidden_states) > 4 and 
                encoder_hidden_states[4] is not None):
                
                try:
                    high_freq_features = encoder_hidden_states[4]
                    # 中间层使用最深层的高频特征
                    if len(high_freq_features) >= 1:
                        target_index = min(index if index is not None else len(high_freq_features)-1, 
                                         len(high_freq_features)-1)
                        current_high_freq_feature = high_freq_features[target_index]
                        
                        # 使用增强条件指导
                        hidden_states = self.condition_guidance(
                            hidden_states=hidden_states,
                            condition_feature=current_high_freq_feature,
                            layer_index=4,  # 中间层的层级索引
                            time_emb=temb,
                            global_condition=high_freq_features[-1]
                        )
                        
                except Exception as e:
                    print(f"Warning: 中间层增强条件指导失败: {e}")
            
            # 时间嵌入
            hidden_states = resnet(hidden_states, temb)

            # 风格注意力
            current_style_feature = encoder_hidden_states[0]
            if len(current_style_feature.shape) == 4:
                batch_size, channel, height, width = current_style_feature.shape
                current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(
                    batch_size, height*width, channel
                )
            hidden_states = style_attn(hidden_states, context=current_style_feature)

        return hidden_states


# 工厂函数：创建增强的UNet块
def get_enhanced_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    channel_attn=False,
    content_channel=32,
    reduction=32,
    enhanced_condition_guidance=True,
    condition_guidance_type="enhanced"
):
    """获取增强的下采样块"""
    
    if down_block_type == "EnhancedMCADownBlock2D":
        return EnhancedMCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            channel_attn=channel_attn,
            content_channel=content_channel,
            reduction=reduction,
            enhanced_condition_guidance=enhanced_condition_guidance,
            condition_guidance_type=condition_guidance_type,
        )
    else:
        # 降级到原始实现
        from .unet_blocks import get_down_block
        return get_down_block(
            down_block_type, num_layers, in_channels, out_channels,
            temb_channels, add_downsample, resnet_eps, resnet_act_fn,
            attn_num_head_channels, resnet_groups, cross_attention_dim,
            downsample_padding, channel_attn, content_channel, reduction
        ) 