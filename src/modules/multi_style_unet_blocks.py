import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .resnet import ResnetBlock2D, Downsample2D, Upsample2D
from .multi_style_attention import MultiStyleSpatialTransformer
from .channel_attn import ChannelAttnBlock


class MultiStyleMCADownBlock2D(nn.Module):
    """支持多风格的MCA下采样块"""
    
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
        attn_num_head_channels: int = 1,
        cross_attention_dim: int = 1280,
        attention_type: str = "default",
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        content_channel: int = 16,
        reduction: int = 32,
    ):
        super().__init__()
        
        resnets = []
        content_attentions = []
        multi_style_attentions = []
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            
            # Content attention blocks
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            
            # ResNet blocks
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
            
            # 🌟 Multi-style attention blocks
            multi_style_attentions.append(
                MultiStyleSpatialTransformer(
                    in_channels=out_channels,
                    n_heads=attn_num_head_channels,
                    d_head=out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    dropout=dropout,
                )
            )
        
        self.content_attentions = nn.ModuleList(content_attentions)
        self.multi_style_attentions = nn.ModuleList(multi_style_attentions)
        self.resnets = nn.ModuleList(resnets)
        
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    in_channels=out_channels, 
                    use_conv=True, 
                    out_channels=out_channels, 
                    padding=downsample_padding, 
                    name="op"
                )
            ])
        else:
            self.downsamplers = None
            
        self.gradient_checkpointing = False
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        index: int,
        temb: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[list] = None
    ):
        """
        Args:
            hidden_states: 输入特征
            index: 层索引
            temb: 时间嵌入
            encoder_hidden_states: [style1_features, content_features, style2_features, style_content_features]
        """
        output_states = ()
        
        # 解包编码器隐藏状态
        if encoder_hidden_states is not None:
            style1_features = encoder_hidden_states[0]  # 第一个风格特征
            content_features = encoder_hidden_states[1]  # 内容特征
            style2_features = encoder_hidden_states[2] if len(encoder_hidden_states) > 2 else style1_features  # 第二个风格特征
        else:
            style1_features = style2_features = content_features = None
        
        for content_attn, resnet, multi_style_attn in zip(
            self.content_attentions, self.resnets, self.multi_style_attentions
        ):
            # Content attention
            if content_features is not None:
                current_content_feature = content_features[index]
                hidden_states = content_attn(hidden_states, current_content_feature)
            
            # Time embedding
            hidden_states = resnet(hidden_states, temb)
            
            # 🌟 Multi-style attention
            if style1_features is not None and style2_features is not None:
                # 准备风格上下文
                style1_context = style1_features
                style2_context = style2_features
                
                # 时间嵌入处理
                time_emb_for_attn = None
                if temb is not None:
                    # 将时间嵌入转换为适合注意力模块的格式
                    time_emb_for_attn = temb
                
                hidden_states = multi_style_attn(
                    hidden_states,
                    style1_context=style1_context,
                    style2_context=style2_context,
                    time_emb=time_emb_for_attn
                )
            
            output_states += (hidden_states,)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        
        return hidden_states, output_states


class MultiStyleUNetMidMCABlock2D(nn.Module):
    """支持多风格的UNet中间MCA块"""
    
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
        attn_num_head_channels: int = 1,
        attention_type: str = "default",
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        content_channel: int = 256,
        reduction: int = 32,
        **kwargs,
    ):
        super().__init__()
        
        resnets = []
        content_attentions = []
        multi_style_attentions = []
        
        # First ResNet block
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
        
        for _ in range(num_layers):
            # Content attention
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            
            # 🌟 Multi-style attention
            multi_style_attentions.append(
                MultiStyleSpatialTransformer(
                    in_channels=in_channels,
                    n_heads=attn_num_head_channels,
                    d_head=in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    dropout=dropout,
                )
            )
            
            # ResNet block
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
        
        self.content_attentions = nn.ModuleList(content_attentions)
        self.multi_style_attentions = nn.ModuleList(multi_style_attentions)
        self.resnets = nn.ModuleList(resnets)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[list] = None,
        index: Optional[int] = None,
    ):
        """
        Args:
            hidden_states: 输入特征
            temb: 时间嵌入
            encoder_hidden_states: [style1_features, content_features, style2_features, style_content_features]
            index: 层索引
        """
        # 第一个ResNet块
        hidden_states = self.resnets[0](hidden_states, temb)
        
        # 解包编码器隐藏状态
        if encoder_hidden_states is not None:
            style1_features = encoder_hidden_states[0]  # 第一个风格特征
            content_features = encoder_hidden_states[1]  # 内容特征
            style2_features = encoder_hidden_states[2] if len(encoder_hidden_states) > 2 else style1_features  # 第二个风格特征
        else:
            style1_features = style2_features = content_features = None
        
        for content_attn, multi_style_attn, resnet in zip(
            self.content_attentions, self.multi_style_attentions, self.resnets[1:]
        ):
            # Content attention
            if content_features is not None and index is not None:
                current_content_feature = content_features[index]
                hidden_states = content_attn(hidden_states, current_content_feature)
            
            # Time embedding
            hidden_states = resnet(hidden_states, temb)
            
            # 🌟 Multi-style attention
            if style1_features is not None and style2_features is not None:
                # 时间嵌入处理
                time_emb_for_attn = None
                if temb is not None:
                    time_emb_for_attn = temb
                
                hidden_states = multi_style_attn(
                    hidden_states,
                    style1_context=style1_features,
                    style2_context=style2_features,
                    time_emb=time_emb_for_attn
                )
        
        return hidden_states


class MultiStyleMCAUpBlock2D(nn.Module):
    """支持多风格的MCA上采样块"""
    
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
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
        attn_num_head_channels: int = 1,
        cross_attention_dim: int = 1280,
        attention_type: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        content_channel: int = 256,
        reduction: int = 32,
    ):
        super().__init__()
        
        resnets = []
        content_attentions = []
        multi_style_attentions = []
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            
            # Content attention
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=resnet_in_channels + res_skip_channels + content_channel,
                    out_channels=resnet_in_channels + res_skip_channels,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            
            # ResNet block
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
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
            
            # 🌟 Multi-style attention
            multi_style_attentions.append(
                MultiStyleSpatialTransformer(
                    in_channels=out_channels,
                    n_heads=attn_num_head_channels,
                    d_head=out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    dropout=dropout,
                )
            )
        
        self.content_attentions = nn.ModuleList(content_attentions)
        self.multi_style_attentions = nn.ModuleList(multi_style_attentions)
        self.resnets = nn.ModuleList(resnets)
        
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple,
        style_structure_features: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[list] = None,
        upsample_size: Optional[tuple] = None,
    ):
        """
        Args:
            hidden_states: 输入特征
            res_hidden_states_tuple: 残差连接特征
            style_structure_features: 风格结构特征
            temb: 时间嵌入
            encoder_hidden_states: [style1_features, content_features, style2_features]
            upsample_size: 上采样尺寸
        """
        # 解包编码器隐藏状态
        if encoder_hidden_states is not None:
            style1_features = encoder_hidden_states[0]  # 第一个风格特征
            content_features = encoder_hidden_states[1]  # 内容特征 
            style2_features = encoder_hidden_states[2] if len(encoder_hidden_states) > 2 else style1_features  # 第二个风格特征
        else:
            style1_features = style2_features = content_features = None
        
        offset_out_sum = 0
        
        for i, (content_attn, resnet, multi_style_attn) in enumerate(zip(
            self.content_attentions, self.resnets, self.multi_style_attentions
        )):
            # 残差连接
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            
            # Content attention
            if style_structure_features is not None:
                hidden_states, offset_out = content_attn(hidden_states, style_structure_features)
                offset_out_sum += offset_out
            else:
                hidden_states = content_attn(hidden_states)
            
            # ResNet block
            hidden_states = resnet(hidden_states, temb)
            
            # 🌟 Multi-style attention
            if style1_features is not None and style2_features is not None:
                # 时间嵌入处理
                time_emb_for_attn = None
                if temb is not None:
                    time_emb_for_attn = temb
                
                hidden_states = multi_style_attn(
                    hidden_states,
                    style1_context=style1_features,
                    style2_context=style2_features,
                    time_emb=time_emb_for_attn
                )
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        
        return hidden_states, offset_out_sum 