"""
V2X-VLM 模型模块

包含:
- V2XVLM: 主模型 (论文 Section 4)
- TrajectoryHead: 轨迹解码头 (论文 Section 4.2)
- FeatureAlignment: 特征对齐模块 (论文 Section 4.3)
"""

from .v2x_vlm import V2XVLM
from .trajectory_head import TrajectoryHead
from .feature_alignment import FeatureAlignment

__all__ = ['V2XVLM', 'TrajectoryHead', 'FeatureAlignment']
