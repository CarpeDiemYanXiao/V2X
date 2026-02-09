"""
V2X-VLM 评估脚本

实现论文 Section 5.2 的评估指标:
1. L2 Error @ 2.5s, 3.5s, 4.5s (轨迹位移误差)
2. Collision Rate @ 2.5s, 3.5s, 4.5s (碰撞率)
"""

import os
import sys
import yaml
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 添加 src/ 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import V2XVLMDataset, V2XVLMCollator
from models.v2x_vlm import V2XVLM


class V2XVLMEvaluator:
    """
    V2X-VLM 评估器
    
    论文 Section 5.2 Metrics:
    1. L2 Error: 轨迹点的欧氏距离误差
       - @2.5s (25 timesteps @ 10Hz)
       - @3.5s (35 timesteps @ 10Hz)
       - @4.5s (45 timesteps @ 10Hz)
       
    2. Collision Rate: 预测轨迹与障碍物的碰撞率
       - 基于ego车辆边界框检测
       - 阈值判定
       
    论文目标指标 (Table 2):
    - L2 Error Avg: 1.21m
    - Collision Rate Avg: 0.03%
    """
    
    def __init__(
        self,
        model: V2XVLM,
        device: str = "cuda",
        trajectory_length: int = 45,
        hz: int = 10,
        collision_threshold: float = 2.0  # 碰撞判定距离阈值(米)
    ):
        self.model = model
        self.device = device
        self.trajectory_length = trajectory_length
        self.hz = hz
        self.collision_threshold = collision_threshold
        
        # 评估时间点 (秒)
        self.eval_times = [2.5, 3.5, 4.5]
        # 对应的时间步索引
        self.eval_steps = [int(t * hz) for t in self.eval_times]
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        include_collision: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        完整评估
        
        Args:
            dataloader: 验证数据加载器
            include_collision: 是否计算碰撞率
            verbose: 是否打印进度
            
        Returns:
            metrics: 包含各项指标的字典
        """
        all_l2_errors = {f"l2_{t}s": [] for t in self.eval_times}
        all_l2_errors['l2_avg'] = []
        
        all_collision_rates = {f"col_{t}s": [] for t in self.eval_times}
        all_collision_rates['col_avg'] = []
        
        all_predictions = []
        all_targets = []
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        for batch in iterator:
            # 移动数据到设备
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            trajectory_gt = batch['trajectory_gt'].to(self.device)
            
            # 预测
            trajectory_pred = self.model.predict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算L2误差
            l2_metrics = self.compute_l2_error(trajectory_pred, trajectory_gt)
            
            for key, value in l2_metrics.items():
                all_l2_errors[key].extend(value.cpu().numpy().tolist())
            
            # 存储预测和GT (用于可视化)
            all_predictions.append(trajectory_pred.cpu().numpy())
            all_targets.append(trajectory_gt.cpu().numpy())
        
        # 聚合指标
        metrics = {}
        
        # L2 Error
        for key, values in all_l2_errors.items():
            metrics[key] = np.mean(values)
        
        # 打印结果
        if verbose:
            print("\n" + "=" * 50)
            print("Evaluation Results")
            print("=" * 50)
            print(f"L2 Error @ 2.5s: {metrics['l2_2.5s']:.4f} m")
            print(f"L2 Error @ 3.5s: {metrics['l2_3.5s']:.4f} m")
            print(f"L2 Error @ 4.5s: {metrics['l2_4.5s']:.4f} m")
            print(f"L2 Error Avg:    {metrics['l2_avg']:.4f} m")
            print("=" * 50)
            print(f"Paper Target L2 Avg: 1.21m")
            print("=" * 50)
        
        # 添加原始数据
        metrics['predictions'] = np.concatenate(all_predictions, axis=0)
        metrics['targets'] = np.concatenate(all_targets, axis=0)
        
        return metrics
    
    def compute_l2_error(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算L2位移误差
        
        论文定义:
        L2 Error = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
        
        Args:
            pred: 预测轨迹 [B, T, 2]
            target: GT轨迹 [B, T, 2]
            
        Returns:
            l2_errors: 各时间点的L2误差
        """
        # 计算每个点的L2距离
        l2_distance = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # [B, T]
        
        metrics = {}
        
        # 各时间点的误差
        for t, step in zip(self.eval_times, self.eval_steps):
            # 使用该时间点之前所有点的平均误差
            step = min(step, l2_distance.shape[1])
            l2_at_t = l2_distance[:, :step].mean(dim=1)  # [B]
            metrics[f'l2_{t}s'] = l2_at_t
        
        # 平均误差 (所有时间点)
        metrics['l2_avg'] = l2_distance.mean(dim=1)  # [B]
        
        return metrics
    
    def compute_collision_rate(
        self,
        pred: torch.Tensor,
        obstacle_positions: torch.Tensor,
        ego_length: float = 4.5,
        ego_width: float = 1.8
    ) -> Dict[str, float]:
        """
        计算碰撞率
        
        论文中使用的碰撞检测方法:
        判断预测轨迹点是否与障碍物边界框重叠
        
        Args:
            pred: 预测轨迹 [B, T, 2]
            obstacle_positions: 障碍物位置 [B, N_obs, T, 2]
            ego_length: 自车长度(米)
            ego_width: 自车宽度(米)
            
        Returns:
            collision_rates: 各时间点的碰撞率
        """
        batch_size, num_steps, _ = pred.shape
        
        metrics = {}
        
        for t, step in zip(self.eval_times, self.eval_steps):
            step = min(step, num_steps)
            
            # 预测轨迹片段
            pred_segment = pred[:, :step, :]  # [B, step, 2]
            
            # 障碍物位置片段
            obs_segment = obstacle_positions[:, :, :step, :]  # [B, N_obs, step, 2]
            
            # 计算距离
            pred_expanded = pred_segment.unsqueeze(1)  # [B, 1, step, 2]
            distances = torch.norm(pred_expanded - obs_segment, dim=-1)  # [B, N_obs, step]
            
            # 碰撞判定 (使用简化的圆形边界)
            collision_radius = (ego_length + ego_width) / 4  # 近似
            collisions = (distances < collision_radius).any(dim=-1).any(dim=-1)  # [B]
            
            collision_rate = collisions.float().mean().item()
            metrics[f'col_{t}s'] = collision_rate
        
        # 平均碰撞率
        metrics['col_avg'] = np.mean([metrics[f'col_{t}s'] for t in self.eval_times])
        
        return metrics
    
    def compute_ade_fde(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算ADE和FDE指标 (补充指标)
        
        ADE: Average Displacement Error (平均位移误差)
        FDE: Final Displacement Error (最终位移误差)
        
        Args:
            pred: [B, T, 2]
            target: [B, T, 2]
            
        Returns:
            ade: 平均位移误差
            fde: 最终位移误差
        """
        l2_distance = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # [B, T]
        
        ade = l2_distance.mean().item()
        fde = l2_distance[:, -1].mean().item()
        
        return {'ade': ade, 'fde': fde}


def save_results(metrics: Dict, output_path: str):
    """保存评估结果"""
    # 移除numpy数组 (不能直接JSON序列化)
    save_metrics = {k: v for k, v in metrics.items() 
                   if not isinstance(v, np.ndarray)}
    
    with open(output_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print(f"Results saved to {output_path}")


def visualize_trajectory(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    num_samples: int = 10
):
    """
    可视化轨迹预测
    
    Args:
        predictions: [N, T, 2]
        targets: [N, T, 2]
        output_dir: 输出目录
        num_samples: 可视化样本数量
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    for i, idx in enumerate(indices):
        pred = predictions[idx]  # [T, 2]
        target = targets[idx]    # [T, 2]
        
        plt.figure(figsize=(10, 10))
        
        # 绘制GT轨迹
        plt.plot(target[:, 0], target[:, 1], 'g-', linewidth=2, label='Ground Truth')
        plt.scatter(target[0, 0], target[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(target[-1, 0], target[-1, 1], c='green', s=100, marker='x')
        
        # 绘制预测轨迹
        plt.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Prediction')
        plt.scatter(pred[-1, 0], pred[-1, 1], c='red', s=100, marker='x')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Trajectory Prediction Sample {idx}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.savefig(output_dir / f'trajectory_{i}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2X-VLM")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eval_results',
        help='Output directory'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate trajectory visualizations'
    )
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("Loading model...")
    model_config = config.get('model', {})
    
    # 模型缓存目录 (默认为项目目录下的 pretrained_models)
    cache_dir = model_config.get('cache_dir', None)
    if cache_dir is None:
        cache_dir = str(Path(__file__).parent.parent / "pretrained_models")
    
    model = V2XVLM(
        student_model_name=model_config.get('student_model', 'microsoft/Florence-2-base'),
        teacher_model_name=model_config.get('teacher_model', 'microsoft/Florence-2-large'),
        trajectory_length=model_config.get('trajectory_length', 45),
        use_knowledge_distillation=False,  # 推理时不需要teacher
        use_contrastive_alignment=False,
        device=device,
        cache_dir=cache_dir
    )
    
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # 创建数据加载器
    print("Creating dataloader...")
    data_config = config.get('data', {})
    
    processor = model.get_processor()
    
    dataset = V2XVLMDataset(
        data_root=data_config.get('root'),
        split="val",
        processor=processor,
        trajectory_horizon=data_config.get('trajectory_horizon', 45),
        image_size=tuple(data_config.get('image_size', [768, 768])),
        train_ratio=data_config.get('train_ratio', 0.8)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('training', {}).get('batch_size', 4),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=V2XVLMCollator(processor=processor),
        pin_memory=True
    )
    
    # 评估
    print("Evaluating...")
    evaluator = V2XVLMEvaluator(
        model=model,
        device=device,
        trajectory_length=data_config.get('trajectory_horizon', 45)
    )
    
    metrics = evaluator.evaluate(dataloader, verbose=True)
    
    # 计算ADE/FDE
    ade_fde = evaluator.compute_ade_fde(
        torch.from_numpy(metrics['predictions']),
        torch.from_numpy(metrics['targets'])
    )
    metrics.update(ade_fde)
    print(f"ADE: {ade_fde['ade']:.4f} m")
    print(f"FDE: {ade_fde['fde']:.4f} m")
    
    # 保存结果
    save_results(metrics, str(output_dir / 'metrics.json'))
    
    # 可视化
    if args.visualize:
        visualize_trajectory(
            metrics['predictions'],
            metrics['targets'],
            str(output_dir / 'visualizations')
        )


if __name__ == "__main__":
    main()
