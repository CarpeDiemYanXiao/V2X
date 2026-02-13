"""
V2X-VLM 综合损失函数

论文 Eq.14:
L_total = L_traj + λ₁ × L_align + λ₂ × L_KD

其中:
- L_traj: 轨迹回归损失 (L1 loss) [Eq.8]
- L_align: 对比特征对齐损失 (InfoNCE) [Eq.12]
- L_KD: 知识蒸馏损失 (KL divergence) [Eq.13]

论文参数设置:
- λ₁ = 0.1 (contrastive weight)
- λ₂ = 0.5 (knowledge distillation weight)
- T = 2.0 (KD temperature)
- κ = 0.07 (contrastive temperature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TrajectoryLoss(nn.Module):
    """
    轨迹回归损失
    
    论文 Eq.8:
    L_traj = Σ_t |τ_t - τ*_t|
    
    使用L1 Loss (也称MAE, Mean Absolute Error)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算轨迹L1损失
        
        Args:
            pred: 预测轨迹 [B, T, 2]
            target: GT轨迹 [B, T, 2]
            mask: 有效点掩码 [B, T], 可选
            
        Returns:
            loss: 标量损失值
        """
        # 计算L1损失
        diff = torch.abs(pred - target)  # [B, T, 2]
        
        if mask is not None:
            # 对有效点求平均
            mask = mask.unsqueeze(-1).float()  # [B, T, 1]
            loss = (diff * mask).sum() / mask.sum().clamp(min=1) / 2
        else:
            if self.reduction == 'mean':
                loss = diff.mean()
            elif self.reduction == 'sum':
                loss = diff.sum()
            else:
                loss = diff
        
        return loss


class ContrastiveAlignmentLoss(nn.Module):
    """
    对比特征对齐损失
    
    论文 Eq.12 (InfoNCE loss):
    L_align = -1/(2N) × Σ[log(exp(s_ii/κ) / Σ_j exp(s_ij/κ))]
    
    对齐student视觉编码器和teacher文本表示
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: κ温度参数, 论文使用0.07
        """
        super().__init__()
        self.temperature = temperature
        # 可学习的log温度 (可选)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
    
    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        use_learned_temp: bool = True
    ) -> torch.Tensor:
        """
        计算对称InfoNCE损失
        
        Args:
            z_v: 视觉投影特征 [B, D], 已L2归一化
            z_t: 文本投影特征 [B, D], 已L2归一化
            use_learned_temp: 是否使用可学习温度
            
        Returns:
            loss: 对称InfoNCE损失
        """
        batch_size = z_v.shape[0]
        
        # 确保特征已归一化
        z_v = F.normalize(z_v, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        
        # 获取温度
        if use_learned_temp:
            temperature = torch.exp(self.log_temp)
        else:
            temperature = self.temperature
        
        # 相似度矩阵 s_ij = (z_v^i · z_t^j) / κ
        similarity = torch.matmul(z_v, z_t.T) / temperature  # [B, B]
        
        # 对角线是正样本对
        labels = torch.arange(batch_size, device=z_v.device)
        
        # 对称损失
        # L_v2t: 给定视觉,预测对应文本
        loss_v2t = F.cross_entropy(similarity, labels)
        
        # L_t2v: 给定文本,预测对应视觉
        loss_t2v = F.cross_entropy(similarity.T, labels)
        
        # 对称平均
        loss = (loss_v2t + loss_t2v) / 2
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失
    
    论文 Eq.13:
    L_KD = KL(p_t || p_s) × T^2
    
    其中:
    - p_t = softmax(F_t / T) (teacher软标签)
    - p_s = softmax(F_s / T) (student软预测)
    - T = 2.0 (蒸馏温度)
    """
    
    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: T蒸馏温度, 论文使用2.0
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算KL散度蒸馏损失
        
        Args:
            student_logits: student输出 [B, D] 或 [B, N, D]
            teacher_logits: teacher输出 [B, D] 或 [B, N, D]
            mask: 有效位置掩码
            
        Returns:
            loss: KL散度损失 × T^2
        """
        T = self.temperature
        
        # 如果是序列,先池化
        if student_logits.dim() == 3:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                student_logits = (student_logits * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
                teacher_logits = (teacher_logits * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                student_logits = student_logits.mean(dim=1)
                teacher_logits = teacher_logits.mean(dim=1)
        
        # 软化的概率分布
        student_prob = F.log_softmax(student_logits / T, dim=-1)
        teacher_prob = F.softmax(teacher_logits / T, dim=-1)
        
        # KL散度
        loss = F.kl_div(student_prob, teacher_prob, reduction='batchmean')
        
        # 乘以T^2来平衡梯度幅度
        loss = loss * (T ** 2)
        
        return loss


class V2XVLMLoss(nn.Module):
    """
    V2X-VLM 综合损失函数
    
    论文 Eq.14:
    L_total = L_traj + λ₁ × L_align + λ₂ × L_KD
    
    默认参数 (论文 Table 1):
    - λ₁ = 0.1
    - λ₂ = 0.5
    - κ = 0.07 (contrastive temperature)
    - T = 2.0 (KD temperature)
    """
    
    def __init__(
        self,
        lambda_align: float = 0.1,
        lambda_kd: float = 0.5,
        contrastive_temperature: float = 0.07,
        kd_temperature: float = 2.0,
        use_contrastive: bool = True,
        use_kd: bool = True
    ):
        """
        Args:
            lambda_align: λ₁ 对比对齐损失权重
            lambda_kd: λ₂ 知识蒸馏损失权重
            contrastive_temperature: κ 对比学习温度
            kd_temperature: T 知识蒸馏温度
            use_contrastive: 是否使用对比对齐损失
            use_kd: 是否使用知识蒸馏损失
        """
        super().__init__()
        
        self.lambda_align = lambda_align
        self.lambda_kd = lambda_kd
        self.use_contrastive = use_contrastive
        self.use_kd = use_kd
        
        # 子损失函数
        self.trajectory_loss = TrajectoryLoss()
        
        if use_contrastive:
            self.contrastive_loss = ContrastiveAlignmentLoss(
                temperature=contrastive_temperature
            )
        
        if use_kd:
            self.kd_loss = KnowledgeDistillationLoss(
                temperature=kd_temperature
            )
    
    def forward(
        self,
        trajectory_pred: torch.Tensor,
        trajectory_gt: torch.Tensor,
        z_v: Optional[torch.Tensor] = None,
        z_t: Optional[torch.Tensor] = None,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            trajectory_pred: 预测轨迹 [B, T, 2]
            trajectory_gt: GT轨迹 [B, T, 2]
            z_v: 视觉投影 [B, D] (for contrastive)
            z_t: 文本投影 [B, D] (for contrastive)
            student_features: student特征 [B, N, D] (for KD)
            teacher_features: teacher特征 [B, N, D] (for KD)
            mask: 有效位置掩码
            
        Returns:
            losses: 包含各项损失和总损失的字典
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 轨迹回归损失 L_traj [Eq.8]
        loss_traj = self.trajectory_loss(trajectory_pred, trajectory_gt)
        losses['loss_traj'] = loss_traj
        total_loss = total_loss + loss_traj
        
        # 2. 对比对齐损失 L_align [Eq.12]
        if self.use_contrastive and z_v is not None and z_t is not None:
            loss_align = self.contrastive_loss(z_v, z_t)
            losses['loss_align'] = loss_align
            total_loss = total_loss + self.lambda_align * loss_align
        
        # 3. 知识蒸馏损失 L_KD [Eq.13]
        if self.use_kd and student_features is not None and teacher_features is not None:
            loss_kd = self.kd_loss(student_features, teacher_features, mask)
            losses['loss_kd'] = loss_kd
            total_loss = total_loss + self.lambda_kd * loss_kd
        
        losses['total'] = total_loss
        
        return losses
    
    def from_model_outputs(
        self,
        model_outputs: Dict[str, torch.Tensor],
        trajectory_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        从模型输出直接计算损失
        
        Args:
            model_outputs: 模型forward的输出字典
            trajectory_gt: GT轨迹
            
        Returns:
            losses: 损失字典
        """
        # 如果模型已经计算了损失,直接返回加权和
        if 'losses' in model_outputs:
            existing_losses = model_outputs['losses']
            total = existing_losses.get('loss_traj', 0.0)
            
            if 'loss_vel' in existing_losses:
                total = total + existing_losses['loss_vel']
            
            if 'loss_align' in existing_losses:
                total = total + self.lambda_align * existing_losses['loss_align']
            
            if 'loss_kd' in existing_losses:
                total = total + self.lambda_kd * existing_losses['loss_kd']
            
            return {**existing_losses, 'total': total}
        
        # 否则重新计算
        return self.forward(
            trajectory_pred=model_outputs['trajectory_pred'],
            trajectory_gt=trajectory_gt
        )


class WeightedTrajectoryLoss(nn.Module):
    """
    加权轨迹损失
    
    对不同时间步使用不同权重:
    - 近期轨迹点更重要 (短期规划)
    - 或者远期轨迹点更重要 (长期规划)
    """
    
    def __init__(
        self,
        trajectory_length: int = 45,
        weight_mode: str = 'exponential',
        decay_factor: float = 0.95
    ):
        """
        Args:
            trajectory_length: 轨迹长度
            weight_mode: 'uniform', 'linear', 'exponential'
            decay_factor: 指数衰减因子
        """
        super().__init__()
        
        if weight_mode == 'uniform':
            weights = torch.ones(trajectory_length)
        elif weight_mode == 'linear':
            # 线性衰减: 近期权重更高
            weights = torch.linspace(1.0, 0.5, trajectory_length)
        elif weight_mode == 'exponential':
            # 指数衰减
            weights = decay_factor ** torch.arange(trajectory_length)
        else:
            weights = torch.ones(trajectory_length)
        
        # 归一化
        weights = weights / weights.sum() * trajectory_length
        
        self.register_buffer('weights', weights)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, T, 2]
            target: [B, T, 2]
        """
        diff = torch.abs(pred - target)  # [B, T, 2]
        
        # 应用时间步权重
        weighted_diff = diff * self.weights.view(1, -1, 1)
        
        loss = weighted_diff.mean()
        
        return loss


class CollisionAwareLoss(nn.Module):
    """
    碰撞感知损失
    
    在轨迹损失基础上,额外惩罚可能导致碰撞的预测
    (用于提高碰撞率指标)
    """
    
    def __init__(
        self,
        collision_threshold: float = 2.0,  # 碰撞判定距离阈值(米)
        penalty_weight: float = 1.0
    ):
        super().__init__()
        self.collision_threshold = collision_threshold
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        obstacle_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测轨迹 [B, T, 2]
            target: GT轨迹 [B, T, 2]
            obstacle_positions: 障碍物位置 [B, N_obs, 2]
        """
        # 基础L1损失
        loss_base = F.l1_loss(pred, target)
        
        if obstacle_positions is None:
            return loss_base
        
        # 计算预测轨迹与障碍物的距离
        # pred: [B, T, 2], obstacles: [B, N_obs, 2]
        pred_expanded = pred.unsqueeze(2)  # [B, T, 1, 2]
        obs_expanded = obstacle_positions.unsqueeze(1)  # [B, 1, N_obs, 2]
        
        distances = torch.norm(pred_expanded - obs_expanded, dim=-1)  # [B, T, N_obs]
        min_distances = distances.min(dim=-1)[0]  # [B, T]
        
        # 碰撞惩罚: 距离小于阈值时增加损失
        collision_penalty = F.relu(self.collision_threshold - min_distances)
        loss_collision = collision_penalty.mean()
        
        total_loss = loss_base + self.penalty_weight * loss_collision
        
        return total_loss
