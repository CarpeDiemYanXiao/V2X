# src/data/dataset.py
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class V2XVLMDataset(Dataset):
    def __init__(self, split: str, cfg):
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg.data_root)

        split_file = self.data_root / "split" / f"{split}.txt"
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        frame_id = self.ids[idx]

        # ---------- 1. 读图（保持原始 RGB） ----------
        v_path = self.data_root / "vehicle_images" / f"{frame_id}.jpg"
        i_path = self.data_root / "infrastructure_images" / f"{frame_id}.jpg"

        v_img = Image.open(v_path).convert("RGB")
        i_img = Image.open(i_path).convert("RGB")

        # ---------- 2. 读 prompt ----------
        prompt_path = self.data_root / "text_prompts" / f"{frame_id}.txt"
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        # 检查 prompt 是否为空或过长
        if not prompt:
            raise ValueError(f"Empty prompt found for frame_id: {frame_id}")
        #if len(prompt) > 512:
            #prompt = prompt[:512]


        # ---------- 3. 读 GT trajectory (核心修改) ----------
        traj = np.load(self.data_root / "ground_truth_trajectories" / f"{frame_id}.npy")
        traj = torch.from_numpy(traj).float()  # [45, 2]

        # [修改] 转换为自车坐标系 (平移 + 旋转)
        
        # A. 平移：归零当前位置
        current_pos = traj[0].clone()
        traj_trans = traj - current_pos  # 先平移，把起点变为 (0,0)
        
        # B. 计算车头朝向 (Heading)
        # 使用 t=0 和 t=1 的向量差来确定车头方向
        dx = traj_trans[1, 0] - traj_trans[0, 0]
        dy = traj_trans[1, 1] - traj_trans[0, 1]
        
        # 处理静止情况：如果车辆不动，默认朝向为 0 (或者你需要额外的 heading 数据)
        if torch.abs(dx) < 1e-5 and torch.abs(dy) < 1e-5:
            theta = torch.tensor(0.0)
        else:
            theta = torch.atan2(dy, dx)
        
        # C. 构建旋转矩阵 (将车头转到 X 轴正方向)
        # 我们需要逆时针旋转 -theta 角度
        c = torch.cos(-theta)
        s = torch.sin(-theta)
        rot_mat = torch.tensor([[c, -s], [s, c]])

        # D. 执行旋转: [45, 2] @ [2, 2]
        # 注意：traj_trans 是 (N, 2)，rot_mat 是 (2, 2)，我们要对每个点做旋转
        # 公式：x' = x*c - y*s, y' = x*s + y*c
        traj_rotated = torch.matmul(traj_trans, rot_mat.T)
        
        # E. 检查并过滤异常值
        # 4.5秒内，车辆最多行驶约 150米（120km/h = 33m/s, 4.5s = 150m）
        # 如果超过这个范围，说明数据有问题
        max_reasonable_distance = 200.0  # 给一些余量
        
        # 计算每个点到起点的距离
        distances = torch.norm(traj_rotated, dim=-1)  # [45]
        max_distance = distances.max().item()
        
        # 如果最大距离超过合理范围，这个样本可能有问题
        if max_distance > max_reasonable_distance:
            # 方案1: 截断到合理范围
            scale_factor = max_reasonable_distance / max_distance
            traj_rotated = traj_rotated * scale_factor
            
            # 或者方案2: 直接跳过这个样本（但会导致训练数据减少）
            # raise ValueError(f"Trajectory too long: {max_distance:.2f}m for frame {frame_id}")
        
        # 额外保护：硬截断到[-300, 300]范围
        traj_rotated = torch.clamp(traj_rotated, min=-300.0, max=300.0)
        
        # 现在的 traj_rotated 特性：
        # - 起点 (0, 0)
        # - X轴正方向 = 车头前方 (直行时 X 增大)
        # - Y轴 = 车身左侧 (左转 Y>0, 右转 Y<0)
        # - 数值范围被限制在 [-500, 500] 米以内

        return {
            "vehicle_img": v_img,
            "infra_img": i_img,
            "prompt": prompt,
            "trajectory": traj_rotated  # <--- 喂给模型旋转后的坐标
        }



