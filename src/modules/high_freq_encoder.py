import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)


def sobel_filter(img):
    """使用Sobel算子提取图像的高频信息
    Args:
        img: [B, C, H, W] 输入图像
    Returns:
        高频特征 [B, 1, H, W] - 单通道高频特征
    """
    # 确保输入是正确的格式
    if len(img.shape) != 4:
        raise ValueError(f"Input should be [B, C, H, W], got {img.shape}")
    
    # 检查输入的数值稳定性
    if torch.isnan(img).any() or torch.isinf(img).any():
        print("警告: Sobel输入包含NaN或Inf")
        return torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3], 
                          dtype=img.dtype, device=img.device)
    
    # 首先将图像转换为灰度
    if img.shape[1] == 3:  # RGB图像
        # 使用RGB到灰度的标准转换公式
        gray_img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    else:
        # 如果已经是单通道，直接使用
        gray_img = img
    
    # 定义Sobel卷积核
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                           dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                           dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    
    # 应用Sobel算子
    pad = F.pad(gray_img, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(pad, sobel_x)
    grad_y = F.conv2d(pad, sobel_y)
    
    # 计算梯度幅值
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 添加小数避免数值不稳定
    
    # 更保守的归一化，避免除零
    grad_max = torch.max(grad_magnitude)
    grad_min = torch.min(grad_magnitude)
    
    if grad_max > grad_min + 1e-8:
        grad_magnitude = (grad_magnitude - grad_min) / (grad_max - grad_min + 1e-8)
        grad_magnitude = torch.clamp(grad_magnitude, 0, 1)  # 限制在[0,1]而不是[-1,1]
    else:
        grad_magnitude = torch.zeros_like(grad_magnitude)
    
    # 检查输出的数值稳定性
    if torch.isnan(grad_magnitude).any() or torch.isinf(grad_magnitude).any():
        print("警告: Sobel输出包含NaN或Inf")
        return torch.zeros_like(grad_magnitude)
    
    # 返回单通道结果
    return grad_magnitude


class SN(object):
    def __init__(
        self, 
        num_svs, 
        num_itrs, 
        num_outputs, 
        transpose=False, 
        eps=1e-12
    ):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels

        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


def high_freq_encoder_arch(ch=64, out_channel_multiplier=1, input_nc=1):
    """定义不同分辨率下的高频编码器架构"""
    arch = {}
    
    # 针对80x80输入
    arch[80] = {'in_channels': [input_nc] + [ch*item for item in [1, 2]],
               'out_channels': [item * ch for item in [1, 2, 4]],
               'resolution': [40, 20, 10]}
    
    # 针对96x96输入
    arch[96] = {'in_channels': [input_nc] + [ch*item for item in [1, 2]],
               'out_channels': [item * ch for item in [1, 2, 4]],
               'resolution': [48, 24, 12]}
    
    # 针对128x128输入
    arch[128] = {'in_channels': [input_nc] + [ch*item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'resolution': [64, 32, 16, 8, 4]}
    
    # 针对256x256输入
    arch[256] = {'in_channels': [input_nc] + [ch*item for item in [1, 2, 4, 8, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
                'resolution': [128, 64, 32, 16, 8, 4]}
    
    return arch


class HighFreqEncoder(ModelMixin, ConfigMixin):
    """高频信息编码器，用于从Sobel高频特征中提取多尺度特征"""

    @register_to_config
    def __init__(self, G_ch=64, G_wide=True, resolution=128,
                 G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, output_dim=1, G_fp16=False,
                 G_init='N02', G_param='SN', input_nc=1):
        super(HighFreqEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        # 根据分辨率确定保存哪些层的特征
        if self.resolution == 96 or self.resolution == 80:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        
        self.out_channel_multiplier = 1
        self.arch = high_freq_encoder_arch(self.ch, self.out_channel_multiplier, input_nc)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                               kernel_size=3, padding=1,
                                               num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                               eps=self.SN_eps)
        
        # 构建多层下采样块
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                   out_channels=self.arch['out_channels'][index],
                                   which_conv=self.which_conv,
                                   wide=self.G_wide,
                                   activation=self.activation,
                                   preactivation=(index > 0),
                                   downsample=nn.AvgPool2d(2))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for High Frequency Encoder initialized parameters: %d' % self.param_count)

    def forward(self, x):
        """前向传播，提取多尺度高频特征
        
        Args:
            x: 输入的高频特征 [B, C, H, W]
            
        Returns:
            最终特征和多尺度特征列表（与内容编码器完全一致）
        """
        h = x
        multi_scale_features = []
        # 🎯 与内容编码器完全一致：首先保存输入特征
        multi_scale_features.append(h)  # high_freq[0]: 原始高频图像
        
        # 通过各个层并保存中间特征
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
            
            # 保存编码特征，与内容编码器一致
            if index in self.save_features[:-1]:  # [0,1,2] (排除最后一个)
                multi_scale_features.append(h)  # high_freq[1], high_freq[2], high_freq[3]
        
        # 🔥 关键修改：与内容编码器完全一致，将最终特征append到列表中
        multi_scale_features.append(h)  # high_freq[4] = h (最终特征，与high_freq[3]相同)
        
        # 返回最终特征和多尺度特征列表
        # 现在multi_scale_features: [原始(1), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
        # 与内容编码器完全对应: [原始(3), 第1层(64), 第2层(128), 第3层(256), 最终特征(256)]
        return h, multi_scale_features


def power_iteration(W, u_, update=True, eps=1e-12):
    """SVD幂迭代算法，用于谱归一化"""
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(v, eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(u, eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


def build_high_freq_encoder(args):
    """构建高频编码器实例
    
    Args:
        args: 参数对象，包含模型配置
        
    Returns:
        高频编码器实例
    """
    high_freq_encoder = HighFreqEncoder(
        G_ch=args.content_start_channel,  # 使用与内容编码器相同的通道基数
        resolution=args.content_image_size[0],
        input_nc=1  # 高频特征输入通常是单通道(梯度幅值)
    )
    print("创建了高频编码器 (High Frequency Encoder)!")
    return high_freq_encoder 