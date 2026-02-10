"""
V2X-VLM 数据集类

实现论文描述的数据加载流程:
- 车端图像 I_v ∈ R^{H_v × W_v × 3}
- 路侧图像 I_i ∈ R^{H_i × W_i × 3}  
- 文本提示 E (场景描述 + 自车位置 + 任务描述)
- GT轨迹 τ* = {(x_t, y_t) | t=1,...,45}
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class V2XVLMDataset(Dataset):
    """
    V2X-VLM 数据集
    
    论文 Section 3 Problem Formulation:
    - I_v: vehicle camera input
    - I_i: infrastructure camera input  
    - E: textual prompt
    - τ*: ground-truth trajectory
    
    论文 Section 4.1:
    图像拼接: [I_v, I_i] ∈ R^{H × (W_v + W_i) × 3}
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        processor = None,
        trajectory_horizon: int = 45,
        image_size: Tuple[int, int] = (768, 768),
        train_ratio: float = 0.8,
        use_scene_descriptions: bool = True
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 'train' 或 'val'
            processor: Florence-2 processor
            trajectory_horizon: 预测时间步数
            image_size: 输入图像尺寸
            train_ratio: 训练集比例
            use_scene_descriptions: 是否使用预生成的场景描述
        """
        self.data_root = Path(data_root)
        self.split = split
        self.processor = processor
        self.trajectory_horizon = trajectory_horizon
        self.image_size = image_size
        
        # 图像目录 (解压后的独立目录)
        self.vehicle_image_dir = self.data_root / "cooperative-vehicle-infrastructure-vehicle-side-image"
        self.infra_image_dir = self.data_root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
        
        # 元数据目录
        self.coop_dir = self.data_root / "cooperative-vehicle-infrastructure"
        self.vehicle_side = self.coop_dir / "vehicle-side"
        self.infra_side = self.coop_dir / "infrastructure-side"
        
        # GT轨迹目录
        self.trajectory_dir = self.data_root / "ground_truth_trajectories"
        
        # 场景描述
        self.use_scene_descriptions = use_scene_descriptions
        self.scene_descriptions = self._load_scene_descriptions()
        
        # 加载车端元数据 (用于获取位姿信息)
        self.vehicle_meta = self._load_vehicle_metadata()
        
        # 加载配对数据并筛选有效样本
        self.samples = self._load_valid_samples()
        
        # 划分训练/验证集
        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"[{split}] 加载 {len(self.samples)} 个样本")
    
    def _load_scene_descriptions(self) -> Dict:
        """加载预生成的场景描述"""
        desc_path = self.data_root / "scene_descriptions.json"
        if desc_path.exists() and self.use_scene_descriptions:
            with open(desc_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'vehicle': {}, 'infrastructure': {}}
    
    def _load_vehicle_metadata(self) -> Dict:
        """加载车端帧元数据"""
        vehicle_info_path = self.vehicle_side / "data_info.json"
        
        with open(vehicle_info_path, 'r') as f:
            vehicle_data = json.load(f)
        
        # 构建帧ID到元数据的映射
        meta = {}
        for item in vehicle_data:
            frame_id = item['image_path'].split('/')[-1].replace('.jpg', '')
            meta[frame_id] = item
        
        return meta
    
    def _load_valid_samples(self) -> List[Dict]:
        """
        加载有效的配对样本
        
        有效条件:
        1. 车端图像存在
        2. 路侧图像存在
        3. GT轨迹存在
        """
        coop_info_path = self.coop_dir / "cooperative/data_info.json"
        
        with open(coop_info_path, 'r') as f:
            coop_data = json.load(f)
        
        samples = []
        
        for item in coop_data:
            # 提取帧ID
            v_frame_id = item['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
            i_frame_id = item['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')
            
            # 检查GT轨迹是否存在
            traj_path = self.trajectory_dir / f"{v_frame_id}.npy"
            if not traj_path.exists():
                continue
            
            # 检查图像是否存在
            v_img_path = self.vehicle_image_dir / f"{v_frame_id}.jpg"
            i_img_path = self.infra_image_dir / f"{i_frame_id}.jpg"
            
            if not v_img_path.exists() or not i_img_path.exists():
                continue
            
            samples.append({
                'vehicle_frame_id': v_frame_id,
                'infra_frame_id': i_frame_id,
                'vehicle_image_path': str(v_img_path),
                'infra_image_path': str(i_img_path),
                'trajectory_path': str(traj_path),
                'system_error_offset': item.get('system_error_offset', {'delta_x': 0, 'delta_y': 0})
            })
        
        return samples
    
    def _load_ego_position(self, vehicle_frame_id: str) -> Tuple[float, float, float]:
        """
        加载自车世界坐标位置和航向角
        
        论文 Section 4.1:
        "The ego vehicle's position in the text prompt is represented in the 
        Virtual World Coordinate System provided by the dataset"

        Returns:
            (x, y, yaw) — 世界坐标位置和航向角(弧度)
        """
        meta = self.vehicle_meta.get(vehicle_frame_id, {})
        
        if not meta:
            return (0.0, 0.0, 0.0)
        
        pose_path = self.vehicle_side / meta.get('calib_novatel_to_world_path', '')
        
        if pose_path.exists():
            with open(pose_path, 'r') as f:
                pose = json.load(f)
            x = pose['translation'][0][0]
            y = pose['translation'][1][0]
            # 从旋转矩阵提取航向角 (与 generate_trajectory_gt.py 保持一致)
            rotation = np.array(pose['rotation'])
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            return (x, y, yaw)
        
        return (0.0, 0.0, 0.0)
    
    def _construct_prompt(
        self, 
        vehicle_frame_id: str,
        infra_frame_id: str, 
        ego_position: Tuple[float, float]
    ) -> str:
        """
        构建完整的文本提示
        
        论文 Section 4.1:
        Text prompt E encompasses three key elements:
        1. Scene description - from VLM understanding
        2. Current position of ego vehicle - from GPS/IMU
        3. Explicit planning task description
        """
        # 获取场景描述
        v_desc = self.scene_descriptions.get('vehicle', {}).get(
            vehicle_frame_id,
            "A driving scene from the vehicle's perspective with roads and vehicles."
        )
        
        i_desc = self.scene_descriptions.get('infrastructure', {}).get(
            infra_frame_id,
            "An intersection scene from infrastructure camera showing road layout and traffic."
        )
        
        # 构建提示 (论文格式)
        prompt = f"""Vehicle View: {v_desc}

Infrastructure View: {i_desc}

Current Ego Vehicle Position: x={ego_position[0]:.2f}m, y={ego_position[1]:.2f}m (heading={ego_position[2]:.4f}rad)

Task: Plan the future trajectory for the ego vehicle over the next 4.5 seconds. Output a sequence of 45 waypoints (x, y) at 10Hz in the ego-centric coordinate system."""

        return prompt
    
    def _load_and_process_images(
        self,
        vehicle_image_path: str,
        infra_image_path: str
    ) -> Image.Image:
        """
        加载并拼接图像
        
        论文 Section 4.2:
        "We first concatenate them along the width for an image tensor 
        [I_v, I_i] ∈ R^{H × (W_v + W_i) × 3}"
        """
        # 加载图像
        v_image = Image.open(vehicle_image_path).convert('RGB')
        i_image = Image.open(infra_image_path).convert('RGB')
        
        # 调整尺寸
        v_image = v_image.resize(self.image_size, Image.Resampling.BILINEAR)
        i_image = i_image.resize(self.image_size, Image.Resampling.BILINEAR)
        
        # 沿宽度方向拼接 [I_v, I_i]
        combined_width = self.image_size[0] * 2
        combined_height = self.image_size[1]
        
        combined_image = Image.new('RGB', (combined_width, combined_height))
        combined_image.paste(v_image, (0, 0))
        combined_image.paste(i_image, (self.image_size[0], 0))
        
        return combined_image
    
    def _load_obstacle_positions(
        self, vehicle_frame_id: str, ego_position: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        加载障碍物世界坐标并转换为 ego-centric 坐标系

        坐标变换与 GT 轨迹完全一致 (world_to_ego_centric):
        1. 平移: dx = x_world - x_ego,  dy = y_world - y_ego
        2. 旋转: 使自车朝向为 x 轴正方向

        Args:
            vehicle_frame_id: 车端帧ID (与 label_world 文件名对应)
            ego_position: (x, y, yaw) 自车世界坐标和航向角

        Returns:
            obstacle_positions: [N_obs, 2] ego-centric 坐标系下的障碍物 xy 位置
        """
        label_path = (
            self.coop_dir / "cooperative" / "label_world" / f"{vehicle_frame_id}.json"
        )

        if not label_path.exists():
            return np.zeros((0, 2), dtype=np.float32)

        with open(label_path, 'r') as f:
            labels = json.load(f)

        if not labels:
            return np.zeros((0, 2), dtype=np.float32)

        cx, cy, cyaw = ego_position
        cos_yaw = np.cos(-cyaw)
        sin_yaw = np.sin(-cyaw)

        positions = []
        for obj in labels:
            loc = obj.get('3d_location', {})
            wx = float(loc.get('x', 0))
            wy = float(loc.get('y', 0))
            # 1. 平移到自车中心
            dx = wx - cx
            dy = wy - cy
            # 2. 旋转到自车朝向 (与 generate_trajectory_gt.py 一致)
            ex = dx * cos_yaw - dy * sin_yaw
            ey = dx * sin_yaw + dy * cos_yaw
            positions.append([ex, ey])

        return np.array(positions, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载并拼接图像
        combined_image = self._load_and_process_images(
            sample['vehicle_image_path'],
            sample['infra_image_path']
        )
        
        # 加载GT轨迹
        trajectory = np.load(sample['trajectory_path']).astype(np.float32)
        
        # 获取自车位置
        ego_position = self._load_ego_position(sample['vehicle_frame_id'])
        
        # 构建文本提示
        text_prompt = self._construct_prompt(
            sample['vehicle_frame_id'],
            sample['infra_frame_id'],
            ego_position
        )
        
        # 使用processor处理 (如果提供)
        if self.processor:
            inputs = self.processor(
                text=text_prompt,
                images=combined_image,
                return_tensors="pt"
            )
            # 移除batch维度
            inputs = {k: v.squeeze(0) if hasattr(v, 'squeeze') else v for k, v in inputs.items()}
        else:
            # 返回原始数据 (用于调试)
            inputs = {
                'image': combined_image,
                'text': text_prompt
            }
        
        # 加载障碍物位置 (用于碰撞率 Collision Rate 计算)
        obstacle_positions = self._load_obstacle_positions(
            sample['vehicle_frame_id'], ego_position
        )

        return {
            **inputs,
            'trajectory_gt': torch.from_numpy(trajectory),
            'frame_id': sample['vehicle_frame_id'],
            'ego_position': torch.tensor(ego_position[:2], dtype=torch.float32),  # prompt只用xy
            'obstacle_positions': torch.from_numpy(obstacle_positions),  # [N_obs, 2]
        }


class V2XVLMCollator:
    """
    数据批次整理器
    
    处理变长序列的padding
    """
    
    def __init__(self, processor=None, pad_token_id=0):
        self.processor = processor
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        # 分离不同类型的数据
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # 动态padding input_ids 和 attention_mask
        input_ids_list = [item['input_ids'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]
        
        # 找到最大长度
        max_len = max(ids.size(0) for ids in input_ids_list)
        
        # Pad到相同长度
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
        
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        
        trajectory_gt = torch.stack([item['trajectory_gt'] for item in batch])
        ego_positions = torch.stack([item['ego_position'] for item in batch])
        frame_ids = [item['frame_id'] for item in batch]

        # 处理变长障碍物位置: padding 到批次内最大数量
        obstacle_list = [item['obstacle_positions'] for item in batch]
        max_n_obs = max(obs.size(0) for obs in obstacle_list) if obstacle_list else 0
        max_n_obs = max(max_n_obs, 1)  # 至少保留1个位置以避免空张量

        padded_obstacles = []
        obstacle_masks = []
        for obs in obstacle_list:
            n_obs = obs.size(0)
            pad_n = max_n_obs - n_obs
            if pad_n > 0:
                obs = torch.cat([obs, torch.full((pad_n, 2), 1e6, dtype=obs.dtype)], dim=0)
                mask = torch.cat([torch.ones(n_obs), torch.zeros(pad_n)])
            else:
                mask = torch.ones(n_obs)
            padded_obstacles.append(obs)
            obstacle_masks.append(mask)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'trajectory_gt': trajectory_gt,
            'ego_position': ego_positions,
            'frame_id': frame_ids,
            'obstacle_positions': torch.stack(padded_obstacles),   # [B, max_N_obs, 2]
            'obstacle_mask': torch.stack(obstacle_masks),          # [B, max_N_obs]
        }


def create_dataloaders(
    data_root: str,
    processor,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (768, 768),
    trajectory_horizon: int = 45,
    train_ratio: float = 0.8
):
    """
    创建训练和验证数据加载器
    """
    from torch.utils.data import DataLoader
    
    train_dataset = V2XVLMDataset(
        data_root=data_root,
        split="train",
        processor=processor,
        trajectory_horizon=trajectory_horizon,
        image_size=image_size,
        train_ratio=train_ratio
    )
    
    val_dataset = V2XVLMDataset(
        data_root=data_root,
        split="val",
        processor=processor,
        trajectory_horizon=trajectory_horizon,
        image_size=image_size,
        train_ratio=train_ratio
    )
    
    collator = V2XVLMCollator(processor=processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=False,  # 禁用以避免 CUDA 内存映射冲突
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=False  # 禁用以避免 CUDA 内存映射冲突
    )
    
    return train_loader, val_loader
