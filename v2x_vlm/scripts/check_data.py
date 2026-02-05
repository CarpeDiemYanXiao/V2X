# check_data.py
import torch
import numpy as np
import os
import sys
# 添加项目根目录到路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-base")
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-large")

from src.utils.config import load_config
from src.data.dataset import V2XVLMDataset
from src.utils.tokenizer import TrajectoryTokenizer

cfg_path = os.path.join(ROOT, "configs", "config.yaml")
cfg = load_config(cfg_path)
tokenizer = TrajectoryTokenizer(cfg)

# 2. 加载数据集
dataset = V2XVLMDataset(split="train", cfg=cfg)
data = dataset[0]

# 3. 检查轨迹
traj_coords = data["trajectory"] # [45, 2]
traj_tokens = tokenizer.coords_to_tokens(traj_coords) # [45, 2]

print(f"=== 数据检查 (Frame ID: {dataset.ids[0]}) ===")
print(f"1. 轨迹形状: {traj_coords.shape}")
print(f"2. 起点坐标 (应为 0.0, 0.0): {traj_coords[0].numpy()}")
print(f"3. 起点 Token (应为 512, 512): {traj_tokens[0].numpy()}")
print(f"4. 终点坐标 (检查数值范围): {traj_coords[-1].numpy()}")
print(f"5. 终点 Token: {traj_tokens[-1].numpy()}")

# 检查是否有 Token 被截断 (变成 0 或 1023)
is_clamped = (traj_tokens == 0) | (traj_tokens == 1023)
if is_clamped.any():
    print(f"⚠️ 警告: 有 {is_clamped.sum()} 个点超出了范围被截断！")
else:
    print("✅ 范围正常，无截断。")