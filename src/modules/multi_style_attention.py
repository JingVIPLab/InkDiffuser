import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention, BasicTransformerBlock


class MultiStyleCrossAttention(nn.Module):
    """多风格交叉注意力模块"""
    
    def __init__(
        self, 
        query_dim: int, 
        context_dim: int = None, 
        heads: int = 8, 
        dim_head: int = 64,
        dropout: float = 0.0,
        time_emb_dim: int = 320
    ):
        super().__init__()
        
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        # 🌟 两个独立的风格注意力分支
        self.style1_attn = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        self.style2_attn = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # 🎯 风格间交叉注意力：让两个风格互相学习
        self.style_fusion_attn = CrossAttention(
            query_dim=context_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # 🚀 动态权重生成网络
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(query_dim + time_emb_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim // 2),
            nn.SiLU(),
            nn.Linear(inner_dim // 2, 2),  # 两个风格的权重
            nn.Softmax(dim=-1)
        )
        
        # 🔧 特征增强模块
        self.feature_enhance = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        style1_context: torch.Tensor,
        style2_context: torch.Tensor,
        time_emb: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ):
        """
        Args:
            hidden_states: [B, N, D] - 查询特征
            style1_context: [B, M, D] - 第一个风格上下文
            style2_context: [B, M, D] - 第二个风格上下文  
            time_emb: [B, time_dim] - 时间嵌入
            attention_mask: 注意力掩码
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 🔥 步骤1: 风格间交叉增强
        # 让两个风格互相学习，提取更丰富的风格信息
        enhanced_style1 = self.style_fusion_attn(
            style1_context, 
            style2_context, 
            attention_mask
        )
        enhanced_style2 = self.style_fusion_attn(
            style2_context, 
            style1_context, 
            attention_mask
        )
        
        # 🎯 步骤2: 分别计算与两个增强风格的注意力
        attn1_output = self.style1_attn(
            hidden_states, 
            enhanced_style1, 
            attention_mask
        )
        attn2_output = self.style2_attn(
            hidden_states, 
            enhanced_style2, 
            attention_mask
        )
        
        # 🚀 步骤3: 计算动态融合权重
        if time_emb is not None:
            # 使用全局特征和时间嵌入生成权重
            global_features = hidden_states.mean(dim=1)  # [B, D]
            weight_input = torch.cat([global_features, time_emb], dim=-1)
            dynamic_weights = self.dynamic_weight_net(weight_input)  # [B, 2]
            
            # 扩展权重维度以匹配序列长度
            weight1 = dynamic_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
            weight2 = dynamic_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
        else:
            # 如果没有时间嵌入，使用固定权重
            weight1 = weight2 = 0.5
        
        # 🌟 步骤4: 自适应融合
        fused_output = weight1 * attn1_output + weight2 * attn2_output
        
        # 🔧 步骤5: 特征增强
        enhanced_output = self.feature_enhance(fused_output)
        
        # 残差连接
        return hidden_states + enhanced_output


class MultiStyleSpatialTransformer(nn.Module):
    """多风格空间变换器"""
    
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int = None,
        time_emb_dim: int = 320,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        
        self.proj_in = nn.Linear(in_channels, inner_dim)
        
        # 🌟 多风格变换器层
        self.transformer_blocks = nn.ModuleList([
            MultiStyleTransformerBlock(
                dim=inner_dim,
                n_heads=n_heads,
                d_head=d_head,
                dropout=dropout,
                context_dim=context_dim,
                time_emb_dim=time_emb_dim,
                checkpoint=use_checkpoint,
            )
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Linear(inner_dim, in_channels)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        style1_context: torch.Tensor = None,
        style2_context: torch.Tensor = None,
        time_emb: torch.Tensor = None
    ):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        
        # 规范化和投影
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        hidden_states = self.proj_in(hidden_states)
        
        # 处理风格上下文
        if style1_context is not None and len(style1_context.shape) == 4:
            b, c, h, w = style1_context.shape
            style1_context = style1_context.permute(0, 2, 3, 1).reshape(b, h * w, c)
            
        if style2_context is not None and len(style2_context.shape) == 4:
            b, c, h, w = style2_context.shape
            style2_context = style2_context.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        # 通过变换器层
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                style1_context=style1_context,
                style2_context=style2_context,
                time_emb=time_emb
            )
        
        # 输出投影
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channel).permute(0, 3, 1, 2)
        
        return hidden_states + residual


class MultiStyleTransformerBlock(nn.Module):
    """多风格变换器块"""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int = None,
        time_emb_dim: int = 320,
        gated_ff: bool = True,
        checkpoint: bool = False,
    ):
        super().__init__()
        
        self.attn1 = CrossAttention(  # 自注意力
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        
        # 🌟 多风格交叉注意力
        self.attn2 = MultiStyleCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            time_emb_dim=time_emb_dim
        )
        
        # 前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.checkpoint = checkpoint
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        style1_context: torch.Tensor = None,
        style2_context: torch.Tensor = None,
        time_emb: torch.Tensor = None
    ):
        # 自注意力
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        
        # 多风格交叉注意力
        if style1_context is not None and style2_context is not None:
            hidden_states = self.attn2(
                self.norm2(hidden_states),
                style1_context=style1_context,
                style2_context=style2_context,
                time_emb=time_emb
            ) + hidden_states
        
        # 前馈网络
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        
        return hidden_states


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, dim: int, dim_out: int = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):
    """GELU门控线性单元"""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate) 