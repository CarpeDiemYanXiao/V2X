# V2X-VLM 论文复现方案

## 目录
1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [数据预处理](#3-数据预处理)
4. [模型架构](#4-模型架构)
5. [训练流程](#5-训练流程)
6. [损失函数](#6-损失函数)
7. [评估指标](#7-评估指标)
8. [实现步骤](#8-实现步骤)
9. [关键代码模块](#9-关键代码模块)
10. [调试与优化](#10-调试与优化)

---

## 1. 项目概述

### 1.1 论文核心思想

V2X-VLM是一个基于视觉语言模型(VLM)的端到端车路协同自动驾驶框架，核心创新点：

1. **多视角融合**: 融合车端(Vehicle-side)和路侧(Infrastructure-side)图像
2. **多模态理解**: 结合视觉特征和文本场景描述
3. **对比学习对齐**: 强化视觉-文本特征对齐
4. **知识蒸馏**: 稳定训练过程，提升学生模型性能

### 1.2 论文目标指标

| 指标 | 2.5s | 3.5s | 4.5s | 平均 |
|------|------|------|------|------|
| L2 Error (m) ↓ | 1.09 | 1.12 | 1.42 | **1.21** |
| Collision Rate (%) ↓ | 0.02 | 0.03 | 0.03 | **0.03** |

### 1.3 技术架构

```
输入:
├── 车端图像 I_v (1080 × 1920 × 3)
├── 路侧图像 I_i (1080 × 1920 × 3)  
└── 文本提示 E (场景描述 + 位置 + 任务)
    ↓
┌─────────────────────────────────────┐
│           Florence-2 VLM            │
│  ┌─────────────┬─────────────────┐  │
│  │ Image Encoder│  Text Encoder  │  │
│  │ (DaViT)      │  (BERT-like)   │  │
│  └──────┬──────┴────────┬───────┘  │
│         │               │           │
│         ▼               ▼           │
│    ┌─────────────────────────┐      │
│    │   Multimodal Fusion     │      │
│    │   (Transformer Decoder) │      │
│    └────────────┬────────────┘      │
└─────────────────┼───────────────────┘
                  ▼
          轨迹输出 τ = {(x_t, y_t)}_{t=1}^{45}
```

---

## 2. 环境配置

### 2.1 硬件要求

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) 或更高
- **内存**: 32GB+ RAM
- **存储**: 100GB+ (数据集 + 模型权重)

### 2.2 软件依赖

```bash
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
tqdm>=4.66.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
pyyaml>=6.0
einops>=0.7.0
timm>=0.9.0
```

### 2.3 Florence-2 模型下载

```python
from transformers import AutoProcessor, AutoModelForCausalLM

# Teacher模型 (frozen)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

# Student模型 (trainable)
student_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)
```

---

## 3. 数据预处理

### 3.1 数据集结构 (DAIR-V2X)

```
data/
├── cooperative-vehicle-infrastructure/
│   ├── vehicle-side/
│   │   ├── calib/novatel_to_world/     # ⭐ 自车位姿 (GT轨迹来源)
│   │   └── data_info.json              # 帧元数据 (含batch_id)
│   ├── infrastructure-side/
│   │   └── data_info.json
│   └── cooperative/
│       ├── data_info.json              # 车路配对关系
│       └── label_world/                # 世界坐标3D标注
│
├── cooperative-vehicle-infrastructure-vehicle-side-image/      # 车端图像
└── cooperative-vehicle-infrastructure-infrastructure-side-image/ # 路侧图像
```

### 3.2 Ground Truth 轨迹生成 ⭐ 最关键步骤

**问题**: `ground_truth_trajectories/` 文件夹为空，需要从 `novatel_to_world/` 生成

**轨迹格式**: `[45, 2]` - 45个时间步 × (x, y) 坐标

```python
# scripts/generate_trajectory_gt.py

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_pose(pose_path):
    """加载单帧位姿"""
    with open(pose_path, 'r') as f:
        data = json.load(f)
    x = data['translation'][0][0]
    y = data['translation'][1][0]
    # 提取航向角 (从旋转矩阵)
    rotation = np.array(data['rotation'])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return x, y, yaw

def world_to_ego_centric(trajectory, current_pose):
    """
    将世界坐标轨迹转换为自车中心坐标系
    
    Args:
        trajectory: [(x, y), ...] 世界坐标序列
        current_pose: (x, y, yaw) 当前帧位姿
    
    Returns:
        ego_trajectory: 以当前帧为原点的相对坐标
    """
    cx, cy, cyaw = current_pose
    ego_trajectory = []
    
    for wx, wy in trajectory:
        # 平移到自车中心
        dx = wx - cx
        dy = wy - cy
        
        # 旋转到自车朝向 (使自车朝向为x轴正方向)
        cos_yaw = np.cos(-cyaw)
        sin_yaw = np.sin(-cyaw)
        ex = dx * cos_yaw - dy * sin_yaw
        ey = dx * sin_yaw + dy * cos_yaw
        
        ego_trajectory.append([ex, ey])
    
    return np.array(ego_trajectory)

def generate_trajectories(data_root, output_dir, horizon=45, fps=10):
    """
    生成所有帧的GT轨迹
    
    Args:
        data_root: 数据集根目录
        output_dir: 输出目录
        horizon: 预测时间步数 (45 = 4.5秒)
        fps: 帧率 (10Hz)
    """
    vehicle_side = Path(data_root) / "cooperative-vehicle-infrastructure/vehicle-side"
    novatel_dir = vehicle_side / "calib/novatel_to_world"
    data_info_path = vehicle_side / "data_info.json"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载元数据
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # 按batch_id分组
    batches = {}
    for item in data_info:
        batch_id = item.get('batch_id', 'unknown')
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(item)
    
    # 对每个batch按帧ID排序
    for batch_id in batches:
        batches[batch_id].sort(key=lambda x: x['image_path'].split('/')[-1])
    
    valid_count = 0
    skip_count = 0
    
    for batch_id, frames in tqdm(batches.items(), desc="Processing batches"):
        frame_ids = [f['image_path'].split('/')[-1].replace('.jpg', '') for f in frames]
        
        for i, frame in enumerate(frames):
            frame_id = frame_ids[i]
            
            # 检查是否有足够的未来帧
            if i + horizon > len(frames):
                skip_count += 1
                continue
            
            # 加载当前帧位姿
            current_pose_path = novatel_dir / f"{frame_id}.json"
            if not current_pose_path.exists():
                skip_count += 1
                continue
            
            current_pose = load_pose(current_pose_path)
            
            # 加载未来轨迹
            future_trajectory = []
            valid = True
            
            for j in range(1, horizon + 1):
                future_frame_id = frame_ids[i + j]
                future_pose_path = novatel_dir / f"{future_frame_id}.json"
                
                if not future_pose_path.exists():
                    valid = False
                    break
                
                fx, fy, _ = load_pose(future_pose_path)
                future_trajectory.append([fx, fy])
            
            if not valid:
                skip_count += 1
                continue
            
            # 转换为自车中心坐标系
            ego_trajectory = world_to_ego_centric(future_trajectory, current_pose)
            
            # 保存
            output_path = output_dir / f"{frame_id}.npy"
            np.save(output_path, ego_trajectory.astype(np.float32))
            valid_count += 1
    
    print(f"生成完成: {valid_count} 个有效轨迹, {skip_count} 个跳过")
    return valid_count

if __name__ == "__main__":
    generate_trajectories(
        data_root="data",
        output_dir="data/ground_truth_trajectories",
        horizon=45,
        fps=10
    )
```

### 3.3 文本提示生成

论文使用 Florence-2-large 的 `<DETAILED_CAPTION>` API 生成场景描述：

```python
# scripts/generate_text_prompts.py

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import torch

def generate_scene_descriptions(image_dir, output_path, device="cuda"):
    """
    使用Florence-2生成场景描述
    """
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    model.eval()
    
    image_dir = Path(image_dir)
    descriptions = {}
    
    for img_path in tqdm(list(image_dir.glob("*.jpg"))):
        frame_id = img_path.stem
        
        # 加载图像
        image = Image.open(img_path).convert("RGB")
        
        # 生成描述
        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        description = processor.decode(outputs[0], skip_special_tokens=True)
        descriptions[frame_id] = description
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    print(f"生成 {len(descriptions)} 个场景描述")

def construct_full_prompt(scene_description, ego_position, task="trajectory_planning"):
    """
    构建完整的文本提示
    
    论文提示结构:
    1. 场景描述 (VLM生成)
    2. 自车当前位置 (从novatel_to_world获取)
    3. 规划任务描述
    """
    prompt = f"""Scene Description: {scene_description}

Current Ego Vehicle Position: x={ego_position[0]:.2f}m, y={ego_position[1]:.2f}m

Task: Plan the future trajectory for the ego vehicle over the next 4.5 seconds (45 waypoints at 10Hz). Output the sequence of (x, y) coordinates in the ego-centric coordinate system."""
    
    return prompt
```

### 3.4 数据集类实现

```python
# src/data/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
from pathlib import Path

class V2XVLMDataset(Dataset):
    """
    V2X-VLM 数据集
    
    每个样本包含:
    - vehicle_image: 车端图像
    - infrastructure_image: 路侧图像
    - text_prompt: 文本提示
    - trajectory_gt: GT轨迹 [45, 2]
    """
    
    def __init__(
        self,
        data_root,
        split="train",
        processor=None,
        trajectory_horizon=45,
        image_size=(768, 768)  # Florence-2输入尺寸
    ):
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
        self.scene_descriptions = self._load_scene_descriptions()
        
        # 加载配对数据
        self.samples = self._load_samples()
        
        # 划分训练/验证集 (8:2)
        split_idx = int(len(self.samples) * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
    
    def _load_scene_descriptions(self):
        """加载预生成的场景描述"""
        desc_path = self.data_root / "scene_descriptions.json"
        if desc_path.exists():
            with open(desc_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_samples(self):
        """加载有效的配对样本"""
        coop_info_path = self.coop_dir / "cooperative/data_info.json"
        with open(coop_info_path, 'r') as f:
            coop_data = json.load(f)
        
        # 加载车端元数据 (获取位姿信息)
        vehicle_info_path = self.vehicle_side / "data_info.json"
        with open(vehicle_info_path, 'r') as f:
            vehicle_data = json.load(f)
        
        # 构建帧ID到元数据的映射
        vehicle_meta = {}
        for item in vehicle_data:
            frame_id = item['image_path'].split('/')[-1].replace('.jpg', '')
            vehicle_meta[frame_id] = item
        
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
                'vehicle_meta': vehicle_meta.get(v_frame_id, {})
            })
        
        return samples
    
    def _load_ego_position(self, vehicle_meta):
        """加载自车世界坐标位置"""
        if not vehicle_meta:
            return (0.0, 0.0)
        
        pose_path = self.vehicle_side / vehicle_meta.get(
            'calib_novatel_to_world_path', ''
        )
        if pose_path.exists():
            with open(pose_path, 'r') as f:
                pose = json.load(f)
            x = pose['translation'][0][0]
            y = pose['translation'][1][0]
            return (x, y)
        return (0.0, 0.0)
    
    def _construct_prompt(self, vehicle_frame_id, ego_position):
        """构建完整的文本提示"""
        # 获取场景描述
        scene_desc = self.scene_descriptions.get(
            vehicle_frame_id,
            "A driving scene with vehicles and road infrastructure."
        )
        
        prompt = f"""Scene: {scene_desc}

Ego Position: ({ego_position[0]:.1f}, {ego_position[1]:.1f})

Task: Predict the ego vehicle's future trajectory for the next 4.5 seconds as a sequence of 45 (x, y) waypoints in the ego-centric coordinate system."""
        
        return prompt
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        v_image = Image.open(sample['vehicle_image_path']).convert('RGB')
        i_image = Image.open(sample['infra_image_path']).convert('RGB')
        
        # 拼接图像 (沿宽度方向)
        # 论文: [I_v, I_i] ∈ R^{H × (W_v + W_i) × 3}
        v_image = v_image.resize(self.image_size)
        i_image = i_image.resize(self.image_size)
        
        combined_image = Image.new('RGB', (self.image_size[0] * 2, self.image_size[1]))
        combined_image.paste(v_image, (0, 0))
        combined_image.paste(i_image, (self.image_size[0], 0))
        
        # 加载GT轨迹
        trajectory = np.load(sample['trajectory_path']).astype(np.float32)
        
        # 获取自车位置并构建prompt
        ego_position = self._load_ego_position(sample['vehicle_meta'])
        text_prompt = self._construct_prompt(
            sample['vehicle_frame_id'],
            ego_position
        )
        
        # 使用processor处理
        if self.processor:
            inputs = self.processor(
                text=text_prompt,
                images=combined_image,
                return_tensors="pt"
            )
            # 移除batch维度
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        else:
            inputs = {
                'image': combined_image,
                'text': text_prompt
            }
        
        return {
            **inputs,
            'trajectory_gt': torch.from_numpy(trajectory),
            'frame_id': sample['vehicle_frame_id']
        }
```

---

## 4. 模型架构

### 4.1 整体架构

```python
# src/models/v2x_vlm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

class V2XVLM(nn.Module):
    """
    V2X-VLM: VLM-based End-to-End Cooperative Autonomous Driving
    
    组件:
    1. Florence-2 VLM Backbone (视觉编码器 + 文本编码器 + 多模态融合)
    2. Trajectory Decoder (轨迹解码头)
    3. Contrastive Alignment Module (对比学习对齐)
    """
    
    def __init__(
        self,
        model_name="microsoft/Florence-2-base",
        trajectory_horizon=45,
        hidden_dim=768,
        freeze_vision_encoder=True
    ):
        super().__init__()
        
        self.trajectory_horizon = trajectory_horizon
        self.hidden_dim = hidden_dim
        
        # 加载Florence-2作为backbone
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 冻结视觉编码器 (论文设定)
        if freeze_vision_encoder:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False
        
        # 获取VLM隐藏层维度
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        
        # 轨迹预测头
        self.trajectory_head = TrajectoryHead(
            input_dim=vlm_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=2,  # (x, y)
            num_waypoints=trajectory_horizon
        )
        
        # 对比学习投影层
        self.visual_proj = nn.Linear(vlm_hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(vlm_hidden_dim, hidden_dim)
    
    def get_visual_embedding(self, pixel_values):
        """提取视觉嵌入"""
        # Florence-2视觉编码器输出
        vision_outputs = self.vlm.vision_tower(pixel_values)
        # 池化得到固定长度表示
        visual_embed = vision_outputs.last_hidden_state.mean(dim=1)
        return visual_embed
    
    def get_text_embedding(self, input_ids, attention_mask):
        """提取文本嵌入"""
        # 使用语言模型编码器
        text_outputs = self.vlm.language_model.model.embed_tokens(input_ids)
        # 带mask的平均池化
        mask_expanded = attention_mask.unsqueeze(-1).float()
        text_embed = (text_outputs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        return text_embed
    
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        labels=None,
        return_embeddings=False
    ):
        """
        前向传播
        
        Args:
            pixel_values: 图像张量 [B, C, H, W]
            input_ids: 文本token [B, L]
            attention_mask: 注意力掩码 [B, L]
            labels: 用于语言模型损失的标签
            return_embeddings: 是否返回嵌入 (用于对比学习)
        
        Returns:
            trajectory: 预测轨迹 [B, 45, 2]
            (可选) visual_embed, text_embed
        """
        # 获取VLM输出
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [B, L, D]
        
        # 取序列的平均作为全局表示
        pooled_output = hidden_states.mean(dim=1)  # [B, D]
        
        # 预测轨迹
        trajectory = self.trajectory_head(pooled_output)  # [B, 45, 2]
        
        result = {'trajectory': trajectory}
        
        if return_embeddings:
            visual_embed = self.get_visual_embedding(pixel_values)
            text_embed = self.get_text_embedding(input_ids, attention_mask)
            
            # 投影到对比学习空间
            z = self.visual_proj(visual_embed)  # 视觉嵌入
            h = self.text_proj(text_embed)      # 文本嵌入
            
            result['visual_embed'] = z
            result['text_embed'] = h
        
        if labels is not None:
            # 语言模型损失 (next-token prediction)
            lm_outputs = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            result['lm_loss'] = lm_outputs.loss
        
        return result


class TrajectoryHead(nn.Module):
    """
    轨迹预测头
    
    将VLM隐藏状态解码为未来轨迹序列
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        output_dim=2,
        num_waypoints=45,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        
        # MLP解码器
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_waypoints * output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            trajectory: [B, num_waypoints, 2]
        """
        out = self.decoder(x)  # [B, 45*2]
        trajectory = out.view(-1, self.num_waypoints, 2)
        return trajectory
```

### 4.2 Teacher-Student 知识蒸馏架构

```python
# src/models/distillation.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class V2XVLMWithDistillation(nn.Module):
    """
    带知识蒸馏的V2X-VLM
    
    Teacher: Florence-2-large (frozen)
    Student: Florence-2-base (trainable)
    """
    
    def __init__(
        self,
        student_model_name="microsoft/Florence-2-base",
        teacher_model_name="microsoft/Florence-2-large",
        trajectory_horizon=45,
        temperature=2.0
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Student模型 (可训练)
        self.student = V2XVLM(
            model_name=student_model_name,
            trajectory_horizon=trajectory_horizon,
            freeze_vision_encoder=True
        )
        
        # Teacher模型 (冻结)
        self.teacher = V2XVLM(
            model_name=teacher_model_name,
            trajectory_horizon=trajectory_horizon,
            freeze_vision_encoder=True
        )
        
        # 冻结Teacher的所有参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        前向传播，同时计算Teacher和Student输出
        """
        # Student前向
        student_outputs = self.student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_embeddings=True
        )
        
        # Teacher前向 (不需要梯度)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=False
            )
        
        return {
            'student_trajectory': student_outputs['trajectory'],
            'teacher_trajectory': teacher_outputs['trajectory'],
            'visual_embed': student_outputs.get('visual_embed'),
            'text_embed': student_outputs.get('text_embed'),
            'lm_loss': student_outputs.get('lm_loss')
        }
```

---

## 5. 训练流程

### 5.1 训练配置

```yaml
# configs/train_config.yaml

# 模型配置
model:
  student_name: "microsoft/Florence-2-base"
  teacher_name: "microsoft/Florence-2-large"
  trajectory_horizon: 45
  hidden_dim: 768
  freeze_vision_encoder: true

# 训练配置
training:
  batch_size: 4
  learning_rate: 1.0e-6
  epochs: 10
  warmup_steps: 100
  weight_decay: 0.01
  gradient_clip: 1.0
  
# 损失权重
loss:
  lambda_align: 0.1      # 对比对齐损失权重
  lambda_kd: 0.5         # 知识蒸馏损失权重
  temperature: 2.0       # 蒸馏温度

# 对比学习配置
contrastive:
  temperature: 0.07      # InfoNCE温度 (κ)
  
# 数据配置
data:
  data_root: "data"
  image_size: [768, 768]
  num_workers: 4
```

### 5.2 训练脚本

```python
# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoProcessor
from tqdm import tqdm
import yaml
from pathlib import Path

from data.dataset import V2XVLMDataset
from models.v2x_vlm import V2XVLM
from models.distillation import V2XVLMWithDistillation
from losses import V2XVLMLoss

def train():
    # 加载配置
    with open("configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(
        config['model']['student_name'],
        trust_remote_code=True
    )
    
    # 创建数据集
    train_dataset = V2XVLMDataset(
        data_root=config['data']['data_root'],
        split="train",
        processor=processor,
        trajectory_horizon=config['model']['trajectory_horizon'],
        image_size=tuple(config['data']['image_size'])
    )
    
    val_dataset = V2XVLMDataset(
        data_root=config['data']['data_root'],
        split="val",
        processor=processor,
        trajectory_horizon=config['model']['trajectory_horizon'],
        image_size=tuple(config['data']['image_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    model = V2XVLMWithDistillation(
        student_model_name=config['model']['student_name'],
        teacher_model_name=config['model']['teacher_name'],
        trajectory_horizon=config['model']['trajectory_horizon'],
        temperature=config['loss']['temperature']
    ).to(device)
    
    # 损失函数
    criterion = V2XVLMLoss(
        lambda_align=config['loss']['lambda_align'],
        lambda_kd=config['loss']['lambda_kd'],
        temperature_kd=config['loss']['temperature'],
        temperature_contrastive=config['contrastive']['temperature']
    )
    
    # 优化器 (只优化student参数)
    optimizer = AdamW(
        model.student.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config['training']['warmup_steps']
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # 训练
        model.train()
        model.teacher.eval()  # Teacher始终eval模式
        
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch in pbar:
            # 移动数据到设备
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            trajectory_gt = batch['trajectory_gt'].to(device)
            
            # 前向传播
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算损失
            loss, loss_dict = criterion(
                student_trajectory=outputs['student_trajectory'],
                teacher_trajectory=outputs['teacher_trajectory'],
                trajectory_gt=trajectory_gt,
                visual_embed=outputs['visual_embed'],
                text_embed=outputs['text_embed']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.student.parameters(),
                config['training']['gradient_clip']
            )
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'L2': f"{loss_dict['traj_loss']:.4f}"
            })
        
        avg_train_loss = train_loss / train_steps
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_l2_errors = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                trajectory_gt = batch['trajectory_gt'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss, loss_dict = criterion(
                    student_trajectory=outputs['student_trajectory'],
                    teacher_trajectory=outputs['teacher_trajectory'],
                    trajectory_gt=trajectory_gt,
                    visual_embed=outputs['visual_embed'],
                    text_embed=outputs['text_embed']
                )
                
                val_loss += loss.item()
                
                # 计算L2 error
                l2_error = compute_l2_error(
                    outputs['student_trajectory'],
                    trajectory_gt
                )
                val_l2_errors.append(l2_error)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_l2 = torch.cat(val_l2_errors).mean().item()
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val L2={avg_l2:.4f}m")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_l2': avg_l2
            }, "checkpoints/best_model.pt")
            print(f"  Saved best model!")


def compute_l2_error(pred_trajectory, gt_trajectory):
    """计算L2误差"""
    # pred_trajectory: [B, 45, 2]
    # gt_trajectory: [B, 45, 2]
    l2 = torch.sqrt(((pred_trajectory - gt_trajectory) ** 2).sum(dim=-1))
    return l2.mean(dim=1)  # [B]


if __name__ == "__main__":
    train()
```

---

## 6. 损失函数

### 6.1 综合损失实现

```python
# src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class V2XVLMLoss(nn.Module):
    """
    V2X-VLM综合损失函数
    
    L_total = L_traj + λ₁ * L_align + λ₂ * L_KD
    
    Components:
    1. L_traj: 轨迹预测损失 (L1/L2)
    2. L_align: 对比学习对齐损失 (InfoNCE)
    3. L_KD: 知识蒸馏损失 (KL散度)
    """
    
    def __init__(
        self,
        lambda_align=0.1,
        lambda_kd=0.5,
        temperature_kd=2.0,
        temperature_contrastive=0.07
    ):
        super().__init__()
        
        self.lambda_align = lambda_align
        self.lambda_kd = lambda_kd
        self.T_kd = temperature_kd
        self.T_con = temperature_contrastive
    
    def trajectory_loss(self, pred, gt):
        """
        轨迹预测损失 (平滑L1损失)
        
        Args:
            pred: [B, 45, 2] 预测轨迹
            gt: [B, 45, 2] GT轨迹
        """
        return F.smooth_l1_loss(pred, gt)
    
    def contrastive_alignment_loss(self, z, h):
        """
        对比学习对齐损失 (InfoNCE)
        
        公式:
        L_align = -1/K * Σ log(exp(S_ii) / Σ exp(S_ij))
        
        其中 S_ij = (z_i^T * h_j) / κ
        
        Args:
            z: [B, D] 视觉嵌入 (已L2归一化)
            h: [B, D] 文本嵌入 (已L2归一化)
        """
        # L2归一化
        z = F.normalize(z, p=2, dim=-1)
        h = F.normalize(h, p=2, dim=-1)
        
        # 计算相似度矩阵
        # S_ij = z_i^T * h_j / κ
        similarity = torch.matmul(z, h.T) / self.T_con  # [B, B]
        
        # 正样本是对角线元素 (i=j)
        batch_size = z.size(0)
        labels = torch.arange(batch_size, device=z.device)
        
        # 双向对比损失
        loss_v2t = F.cross_entropy(similarity, labels)
        loss_t2v = F.cross_entropy(similarity.T, labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def knowledge_distillation_loss(self, student_traj, teacher_traj):
        """
        知识蒸馏损失 (KL散度)
        
        公式:
        L_KD = T² * KL(p_S || p_T)
        
        其中 p = softmax(τ/T)
        
        Args:
            student_traj: [B, 45, 2] 学生模型轨迹
            teacher_traj: [B, 45, 2] 教师模型轨迹
        """
        # 将轨迹展平
        student_flat = student_traj.view(student_traj.size(0), -1)  # [B, 90]
        teacher_flat = teacher_traj.view(teacher_traj.size(0), -1)  # [B, 90]
        
        # 温度缩放后的softmax
        student_soft = F.log_softmax(student_flat / self.T_kd, dim=-1)
        teacher_soft = F.softmax(teacher_flat / self.T_kd, dim=-1)
        
        # KL散度
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        )
        
        # 乘以T²补偿梯度缩放
        return self.T_kd ** 2 * kl_loss
    
    def forward(
        self,
        student_trajectory,
        teacher_trajectory,
        trajectory_gt,
        visual_embed=None,
        text_embed=None
    ):
        """
        计算总损失
        
        Returns:
            total_loss: 标量
            loss_dict: 各项损失的字典
        """
        # 1. 轨迹预测损失
        traj_loss = self.trajectory_loss(student_trajectory, trajectory_gt)
        
        # 2. 对比对齐损失
        align_loss = torch.tensor(0.0, device=trajectory_gt.device)
        if visual_embed is not None and text_embed is not None:
            align_loss = self.contrastive_alignment_loss(visual_embed, text_embed)
        
        # 3. 知识蒸馏损失
        kd_loss = self.knowledge_distillation_loss(
            student_trajectory,
            teacher_trajectory
        )
        
        # 总损失
        total_loss = (
            traj_loss +
            self.lambda_align * align_loss +
            self.lambda_kd * kd_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'traj_loss': traj_loss.item(),
            'align_loss': align_loss.item(),
            'kd_loss': kd_loss.item()
        }
        
        return total_loss, loss_dict
```

---

## 7. 评估指标

### 7.1 评估脚本

```python
# src/evaluate.py

import torch
import numpy as np
from tqdm import tqdm

def evaluate_trajectory(model, dataloader, device, fps=10):
    """
    评估轨迹预测性能
    
    Metrics:
    1. L2 Error @ 2.5s, 3.5s, 4.5s, Avg
    2. Collision Rate @ 2.5s, 3.5s, 4.5s, Avg
    """
    model.eval()
    
    # 时间点对应的帧索引 (10Hz)
    time_indices = {
        '2.5s': 25,   # 25帧
        '3.5s': 35,   # 35帧
        '4.5s': 45    # 45帧 (全部)
    }
    
    l2_errors = {k: [] for k in time_indices.keys()}
    all_predictions = []
    all_gts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            trajectory_gt = batch['trajectory_gt'].to(device)
            
            # 预测
            outputs = model.student(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pred_trajectory = outputs['trajectory']
            
            all_predictions.append(pred_trajectory.cpu())
            all_gts.append(trajectory_gt.cpu())
            
            # 计算各时间点的L2误差
            for time_key, idx in time_indices.items():
                # 取前idx帧
                pred_slice = pred_trajectory[:, :idx, :]  # [B, idx, 2]
                gt_slice = trajectory_gt[:, :idx, :]
                
                # 计算最后一帧的L2误差 (位移误差)
                final_pred = pred_slice[:, -1, :]  # [B, 2]
                final_gt = gt_slice[:, -1, :]
                
                l2 = torch.sqrt(((final_pred - final_gt) ** 2).sum(dim=-1))
                l2_errors[time_key].extend(l2.tolist())
    
    # 汇总结果
    results = {}
    for time_key in time_indices.keys():
        results[f'L2_{time_key}'] = np.mean(l2_errors[time_key])
    
    results['L2_avg'] = np.mean([results[f'L2_{k}'] for k in time_indices.keys()])
    
    # 计算碰撞率 (需要场景中其他车辆的位置信息)
    # 这里简化处理，实际需要加载label_world数据
    collision_rate = compute_collision_rate(
        torch.cat(all_predictions),
        torch.cat(all_gts)
    )
    results['collision_rate'] = collision_rate
    
    return results


def compute_collision_rate(predictions, ground_truths):
    """
    计算碰撞率
    
    基于预测轨迹与GT轨迹的偏差估计碰撞风险
    (完整实现需要其他车辆的轨迹信息)
    """
    # 简化版：使用大偏差作为潜在碰撞
    threshold = 3.0  # 米
    
    l2_all = torch.sqrt(((predictions - ground_truths) ** 2).sum(dim=-1))
    max_l2 = l2_all.max(dim=1)[0]  # 每条轨迹的最大偏差
    
    collision_count = (max_l2 > threshold).sum().item()
    total_count = predictions.size(0)
    
    return collision_count / total_count * 100


def evaluate_at_horizons(model, dataloader, device):
    """
    详细评估各时间点性能
    """
    results = evaluate_trajectory(model, dataloader, device)
    
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"\nL2 Error (m):")
    print(f"  2.5s: {results['L2_2.5s']:.4f}")
    print(f"  3.5s: {results['L2_3.5s']:.4f}")
    print(f"  4.5s: {results['L2_4.5s']:.4f}")
    print(f"  Avg:  {results['L2_avg']:.4f}")
    print(f"\nCollision Rate: {results['collision_rate']:.2f}%")
    print("="*50)
    
    return results
```

---

## 8. 实现步骤

### 8.1 完整复现流程

```bash
# Step 1: 环境准备
pip install -r requirements.txt

# Step 2: 数据准备
# 2.1 确认数据集结构
ls data/cooperative-vehicle-infrastructure/
ls data/cooperative-vehicle-infrastructure-vehicle-side-image/
ls data/cooperative-vehicle-infrastructure-infrastructure-side-image/

# 2.2 生成GT轨迹 (最关键步骤!)
python scripts/generate_trajectory_gt.py

# 2.3 生成场景描述 (可选，耗时较长)
python scripts/generate_text_prompts.py

# Step 3: 验证数据
python scripts/verify_data.py

# Step 4: 训练
python src/train.py

# Step 5: 评估
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 8.2 详细任务清单

| 序号 | 任务 | 优先级 | 状态 |
|------|------|--------|------|
| 1 | 生成GT轨迹 (novatel_to_world → .npy) | ⭐⭐⭐ 最高 | ⬜ |
| 2 | 实现Dataset类 | ⭐⭐⭐ | ⬜ |
| 3 | 实现V2XVLM模型 | ⭐⭐⭐ | ⬜ |
| 4 | 实现损失函数 | ⭐⭐ | ⬜ |
| 5 | 实现训练循环 | ⭐⭐ | ⬜ |
| 6 | 生成场景描述文本 | ⭐ | ⬜ |
| 7 | 实现评估脚本 | ⭐⭐ | ⬜ |
| 8 | 调参优化 | ⭐ | ⬜ |

---

## 9. 关键代码模块

### 9.1 项目目录结构

```
v2x_vlm_reproduce/
├── configs/
│   └── train_config.yaml
├── scripts/
│   ├── generate_trajectory_gt.py    # GT轨迹生成
│   ├── generate_text_prompts.py     # 场景描述生成
│   └── verify_data.py               # 数据验证
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py               # 数据集类
│   ├── models/
│   │   ├── __init__.py
│   │   ├── v2x_vlm.py               # 主模型
│   │   └── distillation.py          # 知识蒸馏
│   ├── losses.py                    # 损失函数
│   ├── train.py                     # 训练脚本
│   └── evaluate.py                  # 评估脚本
├── checkpoints/
├── logs/
└── requirements.txt
```

### 9.2 数据验证脚本

```python
# scripts/verify_data.py

from pathlib import Path
import json
import numpy as np

def verify_data_integrity(data_root):
    """验证数据完整性"""
    data_root = Path(data_root)
    
    issues = []
    
    # 1. 检查图像目录
    v_img_dir = data_root / "cooperative-vehicle-infrastructure-vehicle-side-image"
    i_img_dir = data_root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
    
    if not v_img_dir.exists():
        issues.append(f"❌ 车端图像目录不存在: {v_img_dir}")
    else:
        v_count = len(list(v_img_dir.glob("*.jpg")))
        print(f"✅ 车端图像: {v_count} 张")
    
    if not i_img_dir.exists():
        issues.append(f"❌ 路侧图像目录不存在: {i_img_dir}")
    else:
        i_count = len(list(i_img_dir.glob("*.jpg")))
        print(f"✅ 路侧图像: {i_count} 张")
    
    # 2. 检查GT轨迹
    traj_dir = data_root / "ground_truth_trajectories"
    if not traj_dir.exists():
        issues.append(f"❌ GT轨迹目录不存在: {traj_dir}")
        issues.append("   请运行: python scripts/generate_trajectory_gt.py")
    else:
        traj_count = len(list(traj_dir.glob("*.npy")))
        if traj_count == 0:
            issues.append(f"❌ GT轨迹目录为空!")
        else:
            print(f"✅ GT轨迹: {traj_count} 个")
            
            # 验证轨迹格式
            sample = list(traj_dir.glob("*.npy"))[0]
            arr = np.load(sample)
            if arr.shape == (45, 2):
                print(f"✅ 轨迹格式正确: {arr.shape}")
            else:
                issues.append(f"❌ 轨迹格式错误: {arr.shape}, 应为 (45, 2)")
    
    # 3. 检查配对数据
    coop_info = data_root / "cooperative-vehicle-infrastructure/cooperative/data_info.json"
    if coop_info.exists():
        with open(coop_info, 'r') as f:
            coop_data = json.load(f)
        print(f"✅ 配对帧数: {len(coop_data)}")
    else:
        issues.append(f"❌ 配对数据不存在: {coop_info}")
    
    # 4. 检查novatel_to_world (位姿数据)
    pose_dir = data_root / "cooperative-vehicle-infrastructure/vehicle-side/calib/novatel_to_world"
    if pose_dir.exists():
        pose_count = len(list(pose_dir.glob("*.json")))
        print(f"✅ 位姿文件: {pose_count} 个")
    else:
        issues.append(f"❌ 位姿目录不存在: {pose_dir}")
    
    # 汇总
    print("\n" + "="*50)
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(issue)
    else:
        print("✅ 数据验证通过!")
    print("="*50)
    
    return len(issues) == 0


if __name__ == "__main__":
    verify_data_integrity("data")
```

---

## 10. 调试与优化

### 10.1 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| L2误差过大 | GT轨迹未正确生成 | 检查坐标系转换逻辑 |
| 训练损失不下降 | 学习率过大/过小 | 调整lr, 尝试1e-6到1e-5 |
| 显存不足 | batch_size太大 | 减小batch_size, 使用梯度累积 |
| 模型输出不变 | 梯度消失 | 检查冻结层设置 |

### 10.2 超参数调优建议

```python
# 推荐初始配置 (论文设置)
learning_rate = 1e-6
batch_size = 4
lambda_align = 0.1
lambda_kd = 0.5
temperature_kd = 2.0
temperature_contrastive = 0.07
epochs = 10

# 如果L2误差>2m, 尝试:
# 1. 增加epochs到20
# 2. 调整lambda_align到0.2
# 3. 检查数据预处理是否正确
```

### 10.3 性能监控

```python
# 训练时打印关键指标
print(f"Epoch {epoch}: "
      f"Loss={loss:.4f}, "
      f"L2@2.5s={l2_25:.4f}m, "
      f"L2@4.5s={l2_45:.4f}m, "
      f"Align={align_loss:.4f}, "
      f"KD={kd_loss:.4f}")
```

---

## 附录

### A. 坐标系说明

1. **世界坐标系 (World Coordinate)**: DAIR-V2X使用的虚拟世界坐标系，x-y平面平行于地面，z轴向上
2. **自车中心坐标系 (Ego-centric)**: 以当前帧自车位置为原点，自车朝向为x轴正方向

### B. 论文公式索引

| 公式 | 描述 | 位置 |
|------|------|------|
| Eq.1 | 轨迹定义 | Section 3 |
| Eq.5-6 | 对比对齐相似度 | Section 4.2 |
| Eq.10 | KL蒸馏损失 | Section 4.3 |
| Eq.11 | Next-token预测损失 | Section 4.4 |
| Eq.12 | 对比对齐损失 | Section 4.4 |
| Eq.14 | 总损失函数 | Section 4.4 |

### C. 参考资源

- [Florence-2 HuggingFace](https://huggingface.co/microsoft/Florence-2-base)
- [DAIR-V2X Dataset](https://thudair.baai.ac.cn/index)
- [UniV2X Paper](https://arxiv.org/abs/2404.00717)
