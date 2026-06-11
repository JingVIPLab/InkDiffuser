import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)
from .modules.high_freq_encoder import sobel_filter

class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        high_freq_encoder=None,
        high_freq_fusion_type="adaptive",
        use_intelligent_fusion=True,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.high_freq_encoder = high_freq_encoder
        self.use_intelligent_fusion = use_intelligent_fusion
        
        # 添加可学习的高频权重参数
        self.learnable_high_freq_weight = nn.Parameter(torch.tensor(0.1))
    
        # 在模型级别创建融合模块，而不是在UNet内部
        if self.use_intelligent_fusion and high_freq_encoder is not None:
            from .modules.high_freq_fusion import build_fusion_module
            
            # 为内容特征的每个尺度创建融合模块
            # 🎯 现在两个编码器输出完全一致了！
            # content_residual_features: [原始图像(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            # high_freq_features: [原始高频(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            self.content_high_freq_fusion = nn.ModuleDict()
            
            # 为前3个编码特征创建融合模块，并可选择性为最终特征创建融合
            content_channels = [64, 128, 256, 256]  # 对应content[1], content[2], content[3], content[4]
            high_freq_channels = [64, 128, 256, 256]  # high_freq[1], high_freq[2], high_freq[3], high_freq[4]
            
            # 是否融合最终特征的开关
            self.fuse_final_feature = True  # 可设为参数
            fusion_count = 4 if self.fuse_final_feature else 3
            
            for i in range(fusion_count):
                fusion_module = build_fusion_module(
                    content_channels=content_channels[i],
                    high_freq_channels=high_freq_channels[i],
                    fusion_type=high_freq_fusion_type
                )
                self.content_high_freq_fusion[f"scale_{i}"] = fusion_module
                
            print(f"✅ 内容-高频融合模块创建完成，为{fusion_count}个特征尺度创建融合")
            if self.fuse_final_feature:
                print("  📍 包含最终特征(content[4])的融合，使用high_freq[4]")
            print("  🎯 高频编码器现在输出5层特征，与内容编码器完全对应！")
            print(f"  🎛️ 可学习高频权重初始化为: {self.learnable_high_freq_weight.item():.3f}")

    # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                   [0, 0, 0], 
                                   [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    
    def extract_high_freq(self, x):
        # Convert to grayscale if input is RGB
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x
            
        # Move kernels to device
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)
        
        # Apply Sobel filters
        pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(pad, self.sobel_x)
        grad_y = F.conv2d(pad, self.sobel_y)
        
        # Compute gradient magnitude with numerical stability
        high_freq = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # 添加小常数防止sqrt(0)
        
        # 安全的归一化，防止除零错误
        high_freq_min = high_freq.min()
        high_freq_max = high_freq.max()
        
        # 检查是否有有效的梯度变化
        if torch.abs(high_freq_max - high_freq_min) < 1e-8:
            # 如果没有梯度变化，返回零矩阵
            high_freq = torch.zeros_like(high_freq)
        else:
            # 安全归一化到[-1, 1]范围
            high_freq = (high_freq - high_freq_min) / (high_freq_max - high_freq_min + 1e-8) * 2 - 1
        
        # 数值稳定性检查：确保没有NaN或Inf
        high_freq = torch.nan_to_num(high_freq, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Repeat the single channel to match input channels
        high_freq = high_freq.repeat(1, x.shape[1], 1, 1)
        
        return high_freq

    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
        use_high_freq=False,
    ):
        # 提取风格特征
        style_img_feature, _, _ = self.style_encoder(style_images)
    
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # 获取内容特征
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        
        # 获取参考图像的内容特征
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        # 🔥 关键修改：在这里进行内容特征与高频特征的融合
        if use_high_freq and self.high_freq_encoder is not None and self.use_intelligent_fusion:
            # 使用Sobel算子提取高频信息
            content_high_freq = sobel_filter(content_images)
            
            # 通过高频编码器获取多尺度高频特征
            _, high_freq_features = self.high_freq_encoder(content_high_freq)
            
            # 确保时间步是张量格式
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=content_images.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(content_images.device)
            
            # 融合内容特征与高频特征
            enhanced_content_features = []
            
            # 处理特征融合，注意索引对齐
            # 🎯 现在两个编码器输出完全一致了！
            # content_residual_features: [原始图像(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            # high_freq_features: [原始高频(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            
            for i, content_feat in enumerate(content_residual_features):
                if i == 0:
                    # 第0个是原始图像，不进行融合，直接保留
                    enhanced_content_features.append(content_feat)
                elif 1 <= i <= 3 and i < len(high_freq_features) and f"scale_{i-1}" in self.content_high_freq_fusion:
                    # 对前3个编码特征进行融合 (content[1], content[2], content[3])
                    high_freq_feat = high_freq_features[i]  # 🎯 现在索引完全对应！
                    fused_feat = self.content_high_freq_fusion[f"scale_{i-1}"](
                        content_feat, high_freq_feat, timesteps, i-1
                    )
                    # 应用可学习的全局高频权重
                    final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                    enhanced_content_features.append(final_feat)
                elif i == 4 and hasattr(self, 'fuse_final_feature') and self.fuse_final_feature and f"scale_3" in self.content_high_freq_fusion:
                    # 🎯 完美融合：对最终特征进行融合，使用对应的high_freq[4]
                    high_freq_feat = high_freq_features[4]  # 使用high_freq[4]，完美对应！
                    fused_feat = self.content_high_freq_fusion["scale_3"](
                        content_feat, high_freq_feat, timesteps, 3
                    )
                    # 应用可学习的全局高频权重
                    final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                    enhanced_content_features.append(final_feat)
                else:
                    # 保持原始内容特征
                    enhanced_content_features.append(content_feat)
            
            # 用融合后的特征替换原始内容特征
            content_residual_features = enhanced_content_features

        input_hidden_states = [style_img_feature, content_residual_features, 
                               style_hidden_states, style_content_res_features]

        # UNet不再接收高频特征，因为已经融合到内容特征中
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        high_freq_encoder=None,
        high_freq_fusion_type="adaptive",
        use_intelligent_fusion=True,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.high_freq_encoder = high_freq_encoder
        self.use_intelligent_fusion = use_intelligent_fusion
        
        # 添加可学习的高频权重参数
        self.learnable_high_freq_weight = nn.Parameter(torch.tensor(0.1))

        # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                   [0, 0, 0], 
                                   [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    
        # 在模型级别创建融合模块，而不是在UNet内部
        if self.use_intelligent_fusion and high_freq_encoder is not None:
            from .modules.high_freq_fusion import build_fusion_module
            
            # 为内容特征的每个尺度创建融合模块
            # 🎯 现在两个编码器输出完全一致了！
            # content_residual_features: [原始图像(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            # high_freq_features: [原始高频(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            self.content_high_freq_fusion = nn.ModuleDict()
            
            # 为前3个编码特征创建融合模块，并可选择性为最终特征创建融合
            content_channels = [64, 128, 256, 256]  # 对应content[1], content[2], content[3], content[4]
            high_freq_channels = [64, 128, 256, 256]  # high_freq[1], high_freq[2], high_freq[3], high_freq[4]
            
            # 是否融合最终特征的开关
            self.fuse_final_feature = True  # 可设为参数
            fusion_count = 4 if self.fuse_final_feature else 3
            
            for i in range(fusion_count):
                fusion_module = build_fusion_module(
                    content_channels=content_channels[i],
                    high_freq_channels=high_freq_channels[i],
                    fusion_type=high_freq_fusion_type
                )
                self.content_high_freq_fusion[f"scale_{i}"] = fusion_module
                
            print(f"✅ 内容-高频融合模块创建完成，为{fusion_count}个特征尺度创建融合")
            if self.fuse_final_feature:
                print("  📍 包含最终特征(content[4])的融合，使用high_freq[4]")
            print("  🎯 高频编码器现在输出5层特征，与内容编码器完全对应！")
            print(f"  🎛️ 可学习高频权重初始化为: {self.learnable_high_freq_weight.item():.3f}")

    def extract_high_freq(self, x):
        # Same as above but with numerical stability fixes
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x
            
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)
        
        pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(pad, self.sobel_x)
        grad_y = F.conv2d(pad, self.sobel_y)
        
        # Compute gradient magnitude with numerical stability
        high_freq = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # 添加小常数防止sqrt(0)
        
        # 安全的归一化，防止除零错误
        high_freq_min = high_freq.min()
        high_freq_max = high_freq.max()
        
        # 检查是否有有效的梯度变化
        if torch.abs(high_freq_max - high_freq_min) < 1e-8:
            # 如果没有梯度变化，返回零矩阵
            high_freq = torch.zeros_like(high_freq)
        else:
            # 安全归一化到[-1, 1]范围
            high_freq = (high_freq - high_freq_min) / (high_freq_max - high_freq_min + 1e-8) * 2 - 1
        
        # 数值稳定性检查：确保没有NaN或Inf
        high_freq = torch.nan_to_num(high_freq, nan=0.0, posinf=1.0, neginf=-1.0)
        high_freq = high_freq.repeat(1, x.shape[1], 1, 1)
        
        return high_freq
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
        use_high_freq=False,
    ):
        content_images = cond[0]
        style_images = cond[1]

        # 提取风格特征
        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # 获取内容特征
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        
        # 获取参考图像的内容特征
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        # 🔥 关键修改：在这里进行内容特征与高频特征的融合
        if use_high_freq and self.high_freq_encoder is not None and self.use_intelligent_fusion:
            # 使用Sobel算子提取高频信息
            content_high_freq = sobel_filter(content_images)
            
            # 通过高频编码器获取多尺度高频特征
            _, high_freq_features = self.high_freq_encoder(content_high_freq)
            
            # 确保时间步是张量格式
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=content_images.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(content_images.device)
            
            # 融合内容特征与高频特征
            enhanced_content_features = []
            
            # 处理特征融合，注意索引对齐
            # 🎯 现在两个编码器输出完全一致了！
            # content_residual_features: [原始图像(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            # high_freq_features: [原始高频(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
            
            for i, content_feat in enumerate(content_residual_features):
                if i == 0:
                    # 第0个是原始图像，不进行融合，直接保留
                    enhanced_content_features.append(content_feat)
                elif 1 <= i <= 3 and i < len(high_freq_features) and f"scale_{i-1}" in self.content_high_freq_fusion:
                    # 对前3个编码特征进行融合 (content[1], content[2], content[3])
                    high_freq_feat = high_freq_features[i]  # 🎯 现在索引完全对应！
                    fused_feat = self.content_high_freq_fusion[f"scale_{i-1}"](
                        content_feat, high_freq_feat, timesteps, i-1
                    )
                    # 应用可学习的全局高频权重
                    final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                    enhanced_content_features.append(final_feat)
                elif i == 4 and hasattr(self, 'fuse_final_feature') and self.fuse_final_feature and f"scale_3" in self.content_high_freq_fusion:
                    # 🎯 完美融合：对最终特征进行融合，使用对应的high_freq[4]
                    high_freq_feat = high_freq_features[4]  # 使用high_freq[4]，完美对应！
                    fused_feat = self.content_high_freq_fusion["scale_3"](
                        content_feat, high_freq_feat, timesteps, 3
                    )
                    # 应用可学习的全局高频权重
                    final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                    enhanced_content_features.append(final_feat)
                else:
                    # 保持原始内容特征
                    enhanced_content_features.append(content_feat)
            
            # 用融合后的特征替换原始内容特征
            content_residual_features = enhanced_content_features

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred
