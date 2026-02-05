"""
轨迹解码头

论文 Section 4.2 Trajectory Decoder:
"a simple Trajectory Decoder f_traj(·) based on MLP that is trained from scratch"

f_traj(F_multi) → τ = {(x_t, y_t) | t = 1, ..., T}

论文中:
- F_multi ∈ R^{B × N × D} (多模态特征)
- T = 45 (10Hz × 4.5s)
- 输出 τ ∈ R^{B × T × 2}
"""

import torch
import torch.nn as nn
from typing import Optional


class TrajectoryHead(nn.Module):
    """
    轨迹解码器 - 基于MLP的简单结构
    
    论文 Section 4.2:
    "a simple Trajectory Decoder f_traj(·) based on MLP"
    
    架构:
    1. Token聚合 (mean pooling 或 cls token)
    2. MLP layers
    3. 输出 [B, T, 2]
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        trajectory_length: int = 45,
        mlp_hidden_dims: tuple = (512, 256, 128),
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            hidden_dim: 输入特征维度 (Florence-2 base: 768, large: 1024)
            trajectory_length: 轨迹点数量
            mlp_hidden_dims: MLP隐藏层维度
            dropout: Dropout比例
            activation: 激活函数类型
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.trajectory_length = trajectory_length
        
        # 选择激活函数
        activation_fn = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU()
        }.get(activation, nn.GELU())
        
        # 构建MLP层
        layers = []
        in_dim = hidden_dim
        
        for out_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                activation_fn,
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        # 最终输出层: 输出 T*2 维度 (45个点, 每个点x,y)
        layers.append(nn.Linear(in_dim, trajectory_length * 2))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 多模态特征 [B, N, D] 或 [B, D]
            attention_mask: 注意力掩码 [B, N], 可选
            
        Returns:
            trajectory: 预测轨迹 [B, T, 2]
        """
        # 处理序列特征 - 使用加权平均池化
        if features.dim() == 3:
            if attention_mask is not None:
                # 使用attention mask进行加权池化
                mask = attention_mask.unsqueeze(-1).float()
                features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                # 简单均值池化
                features = features.mean(dim=1)  # [B, D]
        
        # MLP前向传播
        output = self.mlp(features)  # [B, T*2]
        
        # 重塑为轨迹格式
        batch_size = output.shape[0]
        trajectory = output.view(batch_size, self.trajectory_length, 2)
        
        return trajectory


class TrajectoryHeadWithAttention(nn.Module):
    """
    带注意力机制的轨迹解码器
    
    增强版本: 使用交叉注意力聚合token特征
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        trajectory_length: int = 45,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.trajectory_length = trajectory_length
        
        # 可学习的轨迹查询
        self.trajectory_queries = nn.Parameter(
            torch.randn(trajectory_length, hidden_dim) * 0.02
        )
        
        # 交叉注意力层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, 2)
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D]
            attention_mask: [B, N]
            
        Returns:
            trajectory: [B, T, 2]
        """
        batch_size = features.shape[0]
        
        # 扩展查询到batch维度
        queries = self.trajectory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 构建key padding mask
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # True表示需要mask
        else:
            key_padding_mask = None
        
        # 交叉注意力解码
        decoded = self.decoder(
            queries,
            features,
            memory_key_padding_mask=key_padding_mask
        )
        
        # 投影到坐标空间
        trajectory = self.output_proj(decoded)
        
        return trajectory
