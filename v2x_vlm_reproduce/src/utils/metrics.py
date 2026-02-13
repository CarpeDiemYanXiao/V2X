"""
评估指标计算

论文 Section 5.2 Metrics:
- L2 Error: 轨迹位移误差
- Collision Rate: 碰撞率
- ADE/FDE: 平均/最终位移误差 (补充)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union


def compute_l2_error(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    eval_times: tuple = (2.5, 3.5, 4.5),
    hz: int = 10
) -> Dict[str, float]:
    """
    计算L2位移误差
    
    论文定义:
    L2 Error = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
    
    Args:
        pred: 预测轨迹 [B, T, 2] 或 [T, 2]
        target: GT轨迹 [B, T, 2] 或 [T, 2]
        eval_times: 评估时间点(秒)
        hz: 采样频率
        
    Returns:
        metrics: L2误差字典
    """
    # 转换为numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 确保是批次格式
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    
    # 计算L2距离
    l2_distance = np.sqrt(((pred - target) ** 2).sum(axis=-1))  # [B, T]
    
    metrics = {}
    
    for t in eval_times:
        step = int(t * hz)
        step = min(step, l2_distance.shape[1])
        
        l2_at_t = l2_distance[:, :step].mean(axis=1).mean()
        metrics[f'l2_{t}s'] = float(l2_at_t)
    
    # 平均误差 — 论文 Table 2: avg 是三个评估时间点的均值
    metrics['l2_avg'] = float(np.mean([metrics[f'l2_{t}s'] for t in eval_times]))
    
    return metrics


def compute_ade_fde(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    计算ADE和FDE
    
    ADE (Average Displacement Error): 所有时间步的平均L2误差
    FDE (Final Displacement Error): 最后一个时间步的L2误差
    
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
        
    Returns:
        ade, fde
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    
    l2_distance = np.sqrt(((pred - target) ** 2).sum(axis=-1))  # [B, T]
    
    ade = float(l2_distance.mean())
    fde = float(l2_distance[:, -1].mean())
    
    return {'ade': ade, 'fde': fde}


def compute_collision_rate(
    pred: Union[torch.Tensor, np.ndarray],
    obstacle_positions: Union[torch.Tensor, np.ndarray],
    ego_size: Tuple[float, float] = (4.5, 1.8),
    obstacle_size: Tuple[float, float] = (4.5, 1.8),
    eval_times: tuple = (2.5, 3.5, 4.5),
    hz: int = 10
) -> Dict[str, float]:
    """
    计算碰撞率

    论文 Table 2: "collision rate defined by oriented bounding-box overlap
    between the predicted ego trajectory and surrounding obstacles"

    简化为圆形碰撞检测:
    - ego 碰撞半径 = ego 对角线 / 2
    - obstacle 碰撞半径 = obstacle 对角线 / 2
    - 碰撞条件: 两圆心距离 < ego_r + obs_r

    注意: 障碍物位置为当前帧标注, 假设静止.
    只在自车附近 (< 50m) 的障碍物参与碰撞检测, 远处障碍物不会真正碰撞.

    Args:
        pred: 预测轨迹 [B, T, 2]
        obstacle_positions: 障碍物位置 [B, N_obs, 2] (ego-centric)
        ego_size: 自车尺寸 (长, 宽) 米
        obstacle_size: 障碍物尺寸 (长, 宽) 米
        eval_times: 评估时间点
        hz: 采样频率

    Returns:
        collision_rates: 各时间点碰撞率
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(obstacle_positions, torch.Tensor):
        obstacle_positions = obstacle_positions.detach().cpu().numpy()

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]

    batch_size, num_steps, _ = pred.shape

    # 碰撞判定: 用两个物体的等效圆半径之和
    # 等效圆半径 ≈ sqrt(L^2 + W^2) / 2 (对角线的一半)
    # 但这太大了, 实际碰撞用更紧凑的估计: (L+W)/4
    ego_r = (ego_size[0] + ego_size[1]) / 4      # ~1.575m
    obs_r = (obstacle_size[0] + obstacle_size[1]) / 4  # ~1.575m
    collision_threshold = ego_r + obs_r            # ~3.15m (两车中心距)

    metrics = {}

    # 处理障碍物维度
    if obstacle_positions.ndim == 3:
        # [B, N_obs, 2] -> 假设障碍物静止, 广播到所有时间步
        # 预先过滤: 只保留距离原点 50m 以内的障碍物
        # 距离原点太远的障碍物 (包括 1e6 padding) 不可能碰撞
        obs_dist_to_origin = np.linalg.norm(obstacle_positions, axis=-1)  # [B, N_obs]
        far_mask = obs_dist_to_origin > 50.0  # 远处的障碍物
        # 将远处障碍物位置设为极大值, 不会触发碰撞
        obstacle_positions = obstacle_positions.copy()
        obstacle_positions[far_mask] = 1e6

        obstacle_positions = np.tile(
            obstacle_positions[:, :, np.newaxis, :],
            (1, 1, num_steps, 1)
        )

    for t in eval_times:
        step = int(t * hz)
        step = min(step, num_steps)

        # 预测轨迹片段
        pred_segment = pred[:, :step, :]  # [B, step, 2]
        obs_segment = obstacle_positions[:, :, :step, :]  # [B, N_obs, step, 2]

        # 计算距离: ego 预测位置到每个障碍物的距离
        pred_expanded = pred_segment[:, np.newaxis, :, :]  # [B, 1, step, 2]
        distances = np.linalg.norm(pred_expanded - obs_segment, axis=-1)  # [B, N_obs, step]

        # 碰撞: 任意时间步任意障碍物距离 < 阈值
        collisions = (distances < collision_threshold).any(axis=-1).any(axis=-1)  # [B]

        collision_rate = float(collisions.mean())
        metrics[f'col_{t}s'] = collision_rate

    # 平均碰撞率
    metrics['col_avg'] = float(np.mean([metrics[f'col_{t}s'] for t in eval_times]))

    return metrics


def compute_lateral_longitudinal_error(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    计算横向和纵向误差
    
    横向误差: 垂直于行驶方向
    纵向误差: 沿行驶方向
    
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    
    # 计算GT轨迹的方向向量
    direction = np.diff(target, axis=1)  # [B, T-1, 2]
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / (direction_norm + 1e-8)
    
    # 填充最后一个方向
    direction = np.concatenate([direction, direction[:, -1:, :]], axis=1)
    
    # 计算误差向量
    error = pred - target  # [B, T, 2]
    
    # 纵向误差 (沿方向的投影)
    longitudinal = np.abs((error * direction).sum(axis=-1))  # [B, T]
    
    # 横向误差 (垂直于方向)
    lateral_dir = np.stack([-direction[:, :, 1], direction[:, :, 0]], axis=-1)
    lateral = np.abs((error * lateral_dir).sum(axis=-1))  # [B, T]
    
    return {
        'lateral_error': float(lateral.mean()),
        'longitudinal_error': float(longitudinal.mean())
    }


def compute_heading_error(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    计算航向误差 (弧度)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    
    # 计算方向
    pred_dir = np.diff(pred, axis=1)
    target_dir = np.diff(target, axis=1)
    
    # 计算角度
    pred_angle = np.arctan2(pred_dir[:, :, 1], pred_dir[:, :, 0])
    target_angle = np.arctan2(target_dir[:, :, 1], target_dir[:, :, 0])
    
    # 角度差 (处理周期性)
    angle_diff = np.abs(pred_angle - target_angle)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    return float(np.degrees(angle_diff.mean()))


class MetricsTracker:
    """
    指标追踪器
    
    用于训练过程中跟踪和记录指标
    """
    
    def __init__(self):
        self.metrics_history = {}
    
    def update(self, metrics: Dict[str, float], step: int = None):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append({
                'value': value,
                'step': step
            })
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """获取最优值"""
        if metric_name not in self.metrics_history:
            return None, None
        
        values = [item['value'] for item in self.metrics_history[metric_name]]
        
        if mode == 'min':
            idx = np.argmin(values)
        else:
            idx = np.argmax(values)
        
        return values[idx], idx
    
    def get_latest(self, metric_name: str) -> float:
        """获取最新值"""
        if metric_name not in self.metrics_history:
            return None
        return self.metrics_history[metric_name][-1]['value']
    
    def get_average(self, metric_name: str, window: int = 10) -> float:
        """获取滑动平均"""
        if metric_name not in self.metrics_history:
            return None
        values = [item['value'] for item in self.metrics_history[metric_name][-window:]]
        return float(np.mean(values))
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {k: [item['value'] for item in v] for k, v in self.metrics_history.items()}
