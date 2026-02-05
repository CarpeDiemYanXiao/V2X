"""
特征对齐模块

论文 Section 4.3 Contrastive Feature Alignment:
"we design a contrastive learning approach to align scene understanding 
from the student's vision encoder with rich semantic representations 
from the teacher model"

核心公式:
- 视觉投影: z_v = g_v(f_v(I))  [Eq.9]
- 文本投影: z_t = g_t(f_t(E))  [Eq.10]  
- 相似度: s_ij = (z_v^i · z_t^j) / (|z_v^i| |z_t^j|)  [Eq.11]
- InfoNCE: L_align = -1/(2N) * Σ[log(softmax(s_ii/κ))]  [Eq.12]

参数:
- κ = 0.07 (温度参数)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ProjectionHead(nn.Module):
    """
    投影头 g(·)
    
    将编码器特征投影到对齐空间
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.proj = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] 或 [B, N, D]
            
        Returns:
            z: [B, output_dim]
        """
        # 如果是序列,先池化
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        return self.proj(x)


class FeatureAlignment(nn.Module):
    """
    对比特征对齐模块
    
    论文 Section 4.3:
    对齐student视觉编码器和teacher文本编码器的表示
    
    结构:
    - g_v: 视觉投影头
    - g_t: 文本投影头
    - InfoNCE损失计算
    """
    
    def __init__(
        self,
        vision_dim: int = 768,       # Florence-2 base视觉维度
        text_dim: int = 768,          # Florence-2 base文本维度
        projection_dim: int = 256,    # 对齐空间维度
        hidden_dim: int = 512,        # 投影头隐藏维度
        temperature: float = 0.07     # 论文κ参数
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # 视觉投影头 g_v
        self.vision_proj = ProjectionHead(
            input_dim=vision_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
        
        # 文本投影头 g_t  
        self.text_proj = ProjectionHead(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
        
        # 可学习的温度参数 (可选)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        return_similarity: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        投影特征到对齐空间
        
        Args:
            vision_features: [B, D_v] 或 [B, N, D_v] 视觉特征
            text_features: [B, D_t] 或 [B, M, D_t] 文本特征
            return_similarity: 是否返回相似度矩阵
            
        Returns:
            z_v: [B, projection_dim] 视觉投影
            z_t: [B, projection_dim] 文本投影
        """
        # 投影
        z_v = self.vision_proj(vision_features)  # [B, projection_dim]
        z_t = self.text_proj(text_features)       # [B, projection_dim]
        
        # L2归一化
        z_v = F.normalize(z_v, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        
        if return_similarity:
            # 计算相似度矩阵 [Eq.11]
            similarity = torch.matmul(z_v, z_t.T) / self.temperature
            return z_v, z_t, similarity
        
        return z_v, z_t
    
    def compute_loss(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算InfoNCE对比损失
        
        论文 Eq.12:
        L_align = -1/(2N) * Σ[log(exp(s_ii/κ) / Σ_j exp(s_ij/κ))]
        
        Args:
            z_v: [B, D] 归一化的视觉投影
            z_t: [B, D] 归一化的文本投影
            labels: [B] 可选的标签 (用于有监督对比)
            
        Returns:
            loss: 标量损失
        """
        batch_size = z_v.shape[0]
        
        # 获取温度
        temperature = torch.exp(self.log_temperature)
        
        # 计算相似度矩阵 s_ij = (z_v^i · z_t^j) / (|z_v^i| |z_t^j|)
        # 由于已经L2归一化, 直接点积即可
        similarity = torch.matmul(z_v, z_t.T) / temperature  # [B, B]
        
        # 对角线元素是正样本对
        if labels is None:
            labels = torch.arange(batch_size, device=z_v.device)
        
        # 对称InfoNCE损失
        # L_v2t: 视觉→文本
        loss_v2t = F.cross_entropy(similarity, labels)
        
        # L_t2v: 文本→视觉
        loss_t2v = F.cross_entropy(similarity.T, labels)
        
        # 对称平均 [Eq.12]
        loss = (loss_v2t + loss_t2v) / 2
        
        return loss


class CrossModalAlignment(nn.Module):
    """
    跨模态对齐模块 (扩展版)
    
    支持多视图对齐:
    - 车端视觉 ↔ 路侧视觉
    - 融合视觉 ↔ 文本
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # 车端视觉投影
        self.vehicle_proj = ProjectionHead(feature_dim, output_dim=projection_dim)
        
        # 路侧视觉投影
        self.infra_proj = ProjectionHead(feature_dim, output_dim=projection_dim)
        
        # 融合视觉投影
        self.fusion_proj = ProjectionHead(feature_dim, output_dim=projection_dim)
        
        # 文本投影
        self.text_proj = ProjectionHead(feature_dim, output_dim=projection_dim)
    
    def forward(
        self,
        vehicle_features: torch.Tensor,
        infra_features: torch.Tensor,
        fusion_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> dict:
        """
        计算所有对齐损失
        """
        # 投影并归一化
        z_vehicle = F.normalize(self.vehicle_proj(vehicle_features), dim=-1)
        z_infra = F.normalize(self.infra_proj(infra_features), dim=-1)
        z_fusion = F.normalize(self.fusion_proj(fusion_features), dim=-1)
        z_text = F.normalize(self.text_proj(text_features), dim=-1)
        
        # 计算对齐损失
        loss_v2i = self._info_nce_loss(z_vehicle, z_infra)  # 车端-路侧
        loss_f2t = self._info_nce_loss(z_fusion, z_text)    # 融合-文本
        
        return {
            'loss_vehicle_infra': loss_v2i,
            'loss_fusion_text': loss_f2t,
            'total': loss_v2i + loss_f2t
        }
    
    def _info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """对称InfoNCE损失"""
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device)
        
        similarity = torch.matmul(z1, z2.T) / self.temperature
        
        loss_12 = F.cross_entropy(similarity, labels)
        loss_21 = F.cross_entropy(similarity.T, labels)
        
        return (loss_12 + loss_21) / 2
