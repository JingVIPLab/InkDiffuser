import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin

class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """
    FontDiffuser模型 - 支持多风格交叉注意力
    """
    
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        high_freq_encoder=None,
        high_freq_fusion_type="adaptive",
        use_intelligent_fusion=True,
    ):
        super(FontDiffuserModelDPM, self).__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.high_freq_encoder = high_freq_encoder
        self.high_freq_fusion_type = high_freq_fusion_type
        self.use_intelligent_fusion = use_intelligent_fusion
        
        # 🔧 高频特征融合模块
        if high_freq_encoder is not None and use_intelligent_fusion:
            from .modules.high_freq_fusion import build_fusion_module
            
            # 创建融合模块，与标准模型保持一致
            self.content_high_freq_fusion = nn.ModuleDict()
            
            # 正确的内容特征维度：[64, 128, 256, 256] 对应content[1], content[2], content[3], content[4]
            content_channels = [64, 128, 256, 256]
            high_freq_channels = [64, 128, 256, 256]
            
            # 是否融合最终特征的开关
            self.fuse_final_feature = True
            fusion_count = 4 if self.fuse_final_feature else 3
            
            for i in range(fusion_count):
                fusion_module = build_fusion_module(
                    content_channels=content_channels[i],
                    high_freq_channels=high_freq_channels[i],
                    fusion_type=high_freq_fusion_type
                )
                self.content_high_freq_fusion[f"scale_{i}"] = fusion_module
            
            # 可学习的高频权重（一个全局权重）
            self.learnable_high_freq_weight = nn.Parameter(torch.tensor(0.2))
            print(f"✅ 创建了 {fusion_count} 个高频融合模块（与标准模型一致）")
            print(f"🎯 初始高频权重: {self.learnable_high_freq_weight.item():.4f}")
        else:
            self.content_high_freq_fusion = None
            self.learnable_high_freq_weight = None

    def _enhance_content_with_high_freq(self, content_features, high_freq_features, timesteps):
        """
        使用高频特征增强内容特征 - 与标准模型保持一致的融合逻辑
        """
        if (self.high_freq_encoder is None or 
            self.content_high_freq_fusion is None or
            high_freq_features is None):
            return content_features, 0.0
        
        enhanced_features = []
        
        # 确保时间步是张量格式
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=content_features[0].device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(content_features[0].device)
        
        # 处理特征融合，注意索引对齐 - 与标准模型完全一致
        # content_features: [原始图像(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
        # high_freq_features: [原始高频(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
        
        for i, content_feat in enumerate(content_features):
            if i == 0:
                # 第0个是原始图像，不进行融合，直接保留
                enhanced_features.append(content_feat)
            elif 1 <= i <= 3 and i < len(high_freq_features) and f"scale_{i-1}" in self.content_high_freq_fusion:
                # 对前3个编码特征进行融合 (content[1], content[2], content[3])
                high_freq_feat = high_freq_features[i]  # 索引完全对应
                fused_feat = self.content_high_freq_fusion[f"scale_{i-1}"](
                    content_feat, high_freq_feat, timesteps, i-1
                )
                # 应用可学习的全局高频权重
                final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                enhanced_features.append(final_feat)
            elif i == 4 and hasattr(self, 'fuse_final_feature') and self.fuse_final_feature and f"scale_3" in self.content_high_freq_fusion:
                # 对最终特征进行融合，使用对应的high_freq[4]
                high_freq_feat = high_freq_features[4]
                fused_feat = self.content_high_freq_fusion["scale_3"](
                    content_feat, high_freq_feat, timesteps, 3
                )
                # 应用可学习的全局高频权重
                final_feat = content_feat + self.learnable_high_freq_weight * (fused_feat - content_feat)
                enhanced_features.append(final_feat)
            else:
                # 保持原始内容特征
                enhanced_features.append(content_feat)
        
        return enhanced_features, torch.sigmoid(self.learnable_high_freq_weight).item()

    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
        use_high_freq=False,
        use_multi_style=False,  # 🌟 新增多风格支持
    ):
        content_images = cond[0]
        style_images = cond[1]
        
        # 🔍 调试信息：显示cond参数
        print(f"🔍 模型收到参数: use_multi_style={use_multi_style}, len(cond)={len(cond)}")
        
        # 🌟 多风格支持：检查是否有第二个风格图片
        if use_multi_style and len(cond) > 2:
            style_images2 = cond[2]
            # 检查两张图片是否真的不同
            images_equal = torch.equal(style_images, style_images2)
            print(f"🎨 模型：多风格模式，两张风格图片{'相同' if images_equal else '不同'}")
        else:
            style_images2 = style_images  # 单风格模式或只有一张风格图片
            if use_multi_style:
                print(f"ℹ️  模型：多风格模式但只有{len(cond)}个条件参数，将复制使用")
            else:
                print(f"📝 模型：单风格模式")

        # 🌟 提取第一个风格特征
        style_img_feature1, _, style_residual_features1 = self.style_encoder(style_images)
        batch_size, channel, height, width = style_img_feature1.shape
        style_hidden_states1 = style_img_feature1.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # 🌟 提取第二个风格特征
        if use_multi_style and not torch.equal(style_images, style_images2):
            style_img_feature2, _, style_residual_features2 = self.style_encoder(style_images2)
            style_hidden_states2 = style_img_feature2.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        else:
            # 单风格模式：使用相同的风格特征
            style_img_feature2 = style_img_feature1
            style_hidden_states2 = style_hidden_states1
            style_residual_features2 = style_residual_features1

        # 获取内容特征
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        
        # 获取参考图像的内容特征（使用第一张风格图片）
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)
        
        # 🔧 高频特征处理
        if use_high_freq and self.high_freq_encoder is not None:
            try:
                # 🔥 修复：首先使用Sobel算子提取高频信息（RGB转灰度高频）
                from .modules.high_freq_encoder import sobel_filter
                content_high_freq = sobel_filter(content_images)
                
                # 提取高频特征（现在是1-通道输入）
                _, high_freq_features = self.high_freq_encoder(content_high_freq)
                
                # 使用高频特征增强内容特征
                enhanced_content_features, actual_weight = self._enhance_content_with_high_freq(
                    content_residual_features, high_freq_features, timesteps
                )
                
                # 用融合后的特征替换原始内容特征
                content_residual_features = enhanced_content_features
                
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                    if self._debug_counter % 100 == 0:  # 每100步打印一次
                        print(f"🎯 步骤 {self._debug_counter}: 高频权重 = {actual_weight:.4f}")
                else:
                    self._debug_counter = 1
                    print(f"🎯 高频融合权重: {actual_weight:.4f}")
                    
            except Exception as e:
                print(f"⚠️  高频特征处理出错: {e}")
                print(f"   继续使用原始内容特征")

        # 🌟 构建多风格输入状态
        # 检查是否真的有不同的风格图片
        has_different_styles = use_multi_style and len(cond) > 2 and not torch.equal(style_images, style_images2)
        
        if has_different_styles:
            # 真正的多风格模式：传递两个不同的风格特征
            input_hidden_states = [
                style_img_feature1,           # 第一个风格图片特征
                content_residual_features,    # 内容特征
                style_hidden_states1,         # 第一个风格隐藏状态
                style_content_res_features,   # 风格内容特征
                style_img_feature2,           # 第二个风格图片特征
                style_hidden_states2,         # 第二个风格隐藏状态
            ]
            print(f"🌟 模型：多风格输入，6个特征组")
        else:
            # 单风格模式：使用传统的4个特征组
            input_hidden_states = [
                style_img_feature1, 
                content_residual_features, 
                style_hidden_states1, 
                style_content_res_features
            ]
            reason = "单风格模式" if not use_multi_style else "风格图片相同" if use_multi_style and len(cond) > 2 else "只有一个风格参数"
            print(f"🎨 模型：单风格输入，4个特征组（原因：{reason}）")

        # UNet推理
        # 🔧 修复：标准UNet不接受use_multi_style参数，移除该参数
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        
        noise_pred = out[0]
        # 🔧 重要修复：只返回noise_pred，与标准模型保持一致
        # DPM solver期望返回tensor而不是tuple
        return noise_pred


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    
    def __init__(self, content_dim, high_freq_dim, fusion_dim, reduction_ratio=16):
        super().__init__()
        
        # 特征维度对齐
        self.content_proj = nn.Conv2d(content_dim, fusion_dim, 1) if content_dim != fusion_dim else nn.Identity()
        self.high_freq_proj = nn.Conv2d(high_freq_dim, fusion_dim, 1) if high_freq_dim != fusion_dim else nn.Identity()
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_dim * 2, fusion_dim // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim // reduction_ratio, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, 1)
        )
    
    def forward(self, content_feat, high_freq_feat, global_weight):
        # 特征对齐
        content_aligned = self.content_proj(content_feat)
        high_freq_aligned = self.high_freq_proj(high_freq_feat)
        
        # 自适应权重计算
        combined = torch.cat([content_aligned, high_freq_aligned], dim=1)
        weights = self.attention(combined)  # [B, 2, 1, 1]
        
        # 加权融合
        content_weight = weights[:, 0:1]
        high_freq_weight = weights[:, 1:2] * global_weight  # 结合全局权重
        
        fused = content_weight * content_aligned + high_freq_weight * high_freq_aligned
        
        # 特征增强
        enhanced = self.enhance(fused)
        
        return content_feat + enhanced  # 残差连接


class SimpleFeatureFusion(nn.Module):
    """简单特征融合模块"""
    
    def __init__(self, dim):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, content_feat, high_freq_feat, global_weight):
        # 计算局部权重
        combined = torch.cat([content_feat, high_freq_feat], dim=1)
        local_weight = self.weight_net(combined)
        
        # 结合全局和局部权重
        final_weight = global_weight * local_weight
        
        return content_feat + final_weight * high_freq_feat 