"""
V2X-VLM 复现项目主模块
"""

from .data.dataset import V2XVLMDataset, V2XVLMCollator, create_dataloaders
from .models.v2x_vlm import V2XVLM
from .losses.v2x_loss import V2XVLMLoss

__version__ = "1.0.0"

__all__ = [
    'V2XVLMDataset',
    'V2XVLMCollator',
    'create_dataloaders',
    'V2XVLM',
    'V2XVLMLoss'
]
