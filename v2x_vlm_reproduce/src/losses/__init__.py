"""
V2X-VLM 损失函数模块

包含:
- TrajectoryLoss: 轨迹回归损失
- ContrastiveAlignmentLoss: 对比对齐损失
- KnowledgeDistillationLoss: 知识蒸馏损失
- V2XVLMLoss: 综合损失
"""

from .v2x_loss import (
    TrajectoryLoss,
    ContrastiveAlignmentLoss,
    KnowledgeDistillationLoss,
    V2XVLMLoss
)

__all__ = [
    'TrajectoryLoss',
    'ContrastiveAlignmentLoss', 
    'KnowledgeDistillationLoss',
    'V2XVLMLoss'
]
