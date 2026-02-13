"""
V2X-VLM 训练脚本

实现论文 Section 5 的训练配置:
- Optimizer: AdamW
- Learning rate: 1e-6
- Batch size: 4
- Epochs: 10
- LR Scheduler: Linear decay
- Teacher-Student Knowledge Distillation
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# 混合精度支持 (兼容不同PyTorch版本)
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
        GradScaler = None
        autocast = None

# 添加 src/ 目录到路径 (使 data/, models/, losses/ 等模块可导入)
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import V2XVLMDataset, V2XVLMCollator, create_dataloaders
from models.v2x_vlm import V2XVLM
from losses.v2x_loss import V2XVLMLoss
from utils.metrics import compute_l2_error, compute_collision_rate


def setup_logging(output_dir: str) -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger('V2X-VLM')
    logger.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件handler
    log_file = Path(output_dir) / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """
    V2X-VLM 训练器
    
    支持设备: CUDA / NPU (华为昇腾) / CPU
    
    实现完整的训练流程:
    1. 数据加载
    2. 模型前向传播
    3. 损失计算 (L_traj + λ₁×L_align + λ₂×L_KD)
    4. 反向传播和优化
    5. 验证和检查点保存
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: str = "./outputs",
        device: str = "auto"
    ):
        self.config = config
        
        # 创建输出目录 (需要在日志之前)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logging(str(self.output_dir))
        
        # 设备配置: 优先使用配置文件中的设置
        device_config = config.get('device', {})
        config_device = device_config.get('type', 'auto')
        
        if device != "auto":
            # 命令行参数优先级最高
            self.device = device
        elif config_device != "auto":
            # 其次是配置文件
            self.device = config_device
        else:
            # 最后自动检测
            self.device = self._detect_device()
        
        self.logger.info(f"Config loaded")
        self.logger.info(f"Device: {self.device}")
        
        # 初始化模型
        self._init_model()
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 混合精度训练设置 (支持 CUDA 和 NPU)
        self._setup_amp()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _detect_device(self) -> str:
        """自动检测可用设备: NPU > CUDA > CPU"""
        # 尝试 NPU (华为昇腾)
        try:
            import torch_npu
            if torch.npu.is_available():
                npu_count = torch.npu.device_count()
                self.logger.info(f"Detected {npu_count} NPU device(s)")
                return "npu:0"
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"NPU detection failed: {e}")
        
        # 尝试 CUDA
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            self.logger.info(f"Detected {cuda_count} CUDA device(s)")
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            return "cuda:0"
        
        self.logger.warning("No GPU detected, using CPU")
        return "cpu"
    
    def _get_device_type(self) -> str:
        """获取设备类型 (不含设备编号)"""
        if 'npu' in self.device:
            return 'npu'
        elif 'cuda' in self.device:
            return 'cuda'
        return 'cpu'
    
    def _setup_amp(self):
        """设置混合精度训练"""
        train_config = self.config.get('training', {})
        use_amp_config = train_config.get('use_amp', True)
        device_type = self._get_device_type()
        
        # 检查是否支持混合精度
        if not AMP_AVAILABLE:
            self.logger.warning("AMP not available in this PyTorch version")
            self.use_amp = False
            self.scaler = None
            self.amp_dtype = torch.float32
            return
        
        if device_type == 'cpu':
            self.logger.info("AMP disabled for CPU")
            self.use_amp = False
            self.scaler = None
            self.amp_dtype = torch.float32
            return
        
        if device_type == 'npu':
            # NPU 混合精度设置
            try:
                import torch_npu
                self.use_amp = use_amp_config
                if self.use_amp:
                    self.scaler = GradScaler('npu') if hasattr(GradScaler, '__init__') else GradScaler()
                    self.amp_dtype = torch.float16
                    self.logger.info("NPU AMP enabled with float16")
                else:
                    self.scaler = None
                    self.amp_dtype = torch.float32
            except Exception as e:
                self.logger.warning(f"NPU AMP setup failed: {e}, disabling AMP")
                self.use_amp = False
                self.scaler = None
                self.amp_dtype = torch.float32
            return
        
        # CUDA 混合精度设置
        self.use_amp = use_amp_config
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')
            except TypeError:
                # 旧版本 PyTorch
                self.scaler = GradScaler()
            self.amp_dtype = torch.float16
            self.logger.info("CUDA AMP enabled with float16")
        else:
            self.scaler = None
            self.amp_dtype = torch.float32
    
    def _init_model(self):
        """初始化模型"""
        model_config = self.config.get('model', {})
        
        self.logger.info("Initializing V2X-VLM model...")
        
        # 模型缓存目录 (默认为项目目录下的 pretrained_models)
        cache_dir = model_config.get('cache_dir', None)
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / "pretrained_models")
        
        self.model = V2XVLM(
            student_model_name=model_config.get('student_model', 'microsoft/Florence-2-base'),
            teacher_model_name=model_config.get('teacher_model', 'microsoft/Florence-2-large'),
            trajectory_length=model_config.get('trajectory_length', 45),
            hidden_dim=model_config.get('hidden_dim', 768),
            teacher_hidden_dim=model_config.get('teacher_hidden_dim', 1024),
            projection_dim=model_config.get('projection_dim', 256),
            temperature=model_config.get('temperature', 0.07),
            kd_temperature=model_config.get('kd_temperature', 2.0),
            freeze_teacher=True,
            use_knowledge_distillation=model_config.get('use_kd', True),
            use_contrastive_alignment=model_config.get('use_contrastive', True),
            device=self.device,
            cache_dir=cache_dir
        )
        
        self.model = self.model.to(self.device)
        
        # 获取processor
        self.processor = self.model.get_processor()
        
        # 统计参数量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")
    
    def _init_loss(self):
        """初始化损失函数"""
        loss_config = self.config.get('loss', {})
        
        self.criterion = V2XVLMLoss(
            lambda_align=loss_config.get('lambda_align', 0.1),
            lambda_kd=loss_config.get('lambda_kd', 0.5),
            contrastive_temperature=loss_config.get('contrastive_temperature', 0.07),
            kd_temperature=loss_config.get('kd_temperature', 2.0),
            use_contrastive=loss_config.get('use_contrastive', True),
            use_kd=loss_config.get('use_kd', True)
        )
        
        self.logger.info(f"Loss weights: λ_align={loss_config.get('lambda_align', 0.1)}, λ_kd={loss_config.get('lambda_kd', 0.5)}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config.get('training', {})
        
        base_lr = float(train_config.get('learning_rate', 1e-6))
        
        # 3层差分学习率
        param_groups = self.model.get_trainable_parameters(base_lr=base_lr)
        
        # AdamW优化器
        self.optimizer = AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(train_config.get('weight_decay', 0.01)),
            betas=(0.9, 0.999)
        )
        
        # 学习率调度: cosine annealing with warmup
        epochs = train_config.get('epochs', 20)
        scheduler_type = train_config.get('scheduler', 'cosine')
        warmup_epochs = train_config.get('warmup_epochs', 2)
        
        if scheduler_type == 'cosine':
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=1e-8
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        elif scheduler_type == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=epochs
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: AdamW, vision_lr={base_lr*0.1:.1e}, backbone_lr={base_lr}, head_lr={base_lr*100:.1e}")
        self.logger.info(f"Scheduler: {scheduler_type}, warmup={warmup_epochs}, epochs={epochs}")
    
    def create_dataloaders(self):
        """创建数据加载器"""
        data_config = self.config.get('data', {})
        train_config = self.config.get('training', {})
        
        self.logger.info("Creating dataloaders...")
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_root=data_config.get('root'),
            processor=self.processor,
            batch_size=train_config.get('batch_size', 4),
            num_workers=data_config.get('num_workers', 4),
            image_size=tuple(data_config.get('image_size', [768, 768])),
            trajectory_horizon=data_config.get('trajectory_horizon', 45),
            train_ratio=data_config.get('train_ratio', 0.8)
        )
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'traj': 0.0,
            'align': 0.0,
            'kd': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # 移动数据到设备
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            trajectory_gt = batch['trajectory_gt'].to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播 (混合精度)
            if self.use_amp:
                device_type = self._get_device_type()
                with autocast(device_type, dtype=self.amp_dtype):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        trajectory_gt=trajectory_gt
                    )
                    losses = self.criterion.from_model_outputs(outputs, trajectory_gt)
                    loss = losses['total']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('training', {}).get('grad_clip', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    trajectory_gt=trajectory_gt
                )
                losses = self.criterion.from_model_outputs(outputs, trajectory_gt)
                loss = losses['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('training', {}).get('grad_clip', 1.0)
                )
                self.optimizer.step()
            
            # 统计损失
            epoch_losses['total'] += loss.item()
            epoch_losses['traj'] += losses.get('loss_traj', torch.tensor(0.0)).item()
            epoch_losses['align'] += losses.get('loss_align', torch.tensor(0.0)).item()
            epoch_losses['kd'] += losses.get('loss_kd', torch.tensor(0.0)).item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'traj': f"{losses.get('loss_traj', torch.tensor(0.0)).item():.4f}"
            })
            
            self.global_step += 1
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证 —— 输出论文全部 6 项指标"""
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'traj': 0.0,
        }
        all_preds = []
        all_targets = []
        all_obstacles = []
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            trajectory_gt = batch['trajectory_gt'].to(self.device)

            # 前向传播
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                trajectory_gt=trajectory_gt
            )

            losses = self.criterion.from_model_outputs(outputs, trajectory_gt)

            val_losses['total'] += losses['total'].item()
            val_losses['traj'] += losses.get('loss_traj', torch.tensor(0.0)).item()
            num_batches += 1

            # MLP 轨迹预测
            pred_traj = outputs['trajectory_pred']

            # 收集预测和GT
            all_preds.append(pred_traj.cpu())
            all_targets.append(trajectory_gt.cpu())

            # 收集障碍物数据 (已在 Collator 中 padding)
            if 'obstacle_positions' in batch:
                # 根据 mask 只保留有效障碍物
                obs = batch['obstacle_positions']          # [B, max_N, 2]
                obs_mask = batch['obstacle_mask']           # [B, max_N]
                all_obstacles.append((obs, obs_mask))

        # 平均损失
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        # 拼接所有样本
        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        all_targets_np = torch.cat(all_targets, dim=0).numpy()

        # ===== L2 Error @ 2.5s / 3.5s / 4.5s + avg =====
        l2_metrics = compute_l2_error(all_preds_np, all_targets_np)
        val_losses.update(l2_metrics)   # l2_2.5s, l2_3.5s, l2_4.5s, l2_avg

        # ===== Collision Rate @ 2.5s / 3.5s / 4.5s + avg =====
        if all_obstacles:
            # 各批次 max_N_obs 可能不同，需统一 pad 到全局最大值
            global_max_n = max(obs.shape[1] for obs, _ in all_obstacles)
            padded_obs_list = []
            for obs, mask in all_obstacles:
                pad_n = global_max_n - obs.shape[1]
                if pad_n > 0:
                    obs = torch.cat([obs, torch.full((obs.shape[0], pad_n, 2), 1e6, dtype=obs.dtype)], dim=1)
                padded_obs_list.append(obs)
            all_obs_np = torch.cat(padded_obs_list, dim=0).numpy()  # [B_total, global_max_N, 2]
            col_metrics = compute_collision_rate(all_preds_np, all_obs_np)
            val_losses.update(col_metrics)  # col_2.5s, col_3.5s, col_4.5s, col_avg

        return val_losses
    
    def train(self):
        """完整训练流程"""
        epochs = self.config.get('training', {}).get('epochs', 10)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_losses = self.train_epoch()
            self.logger.info(
                f"Epoch {self.current_epoch} Train - "
                f"Total: {train_losses['total']:.4f}, "
                f"Traj: {train_losses['traj']:.4f}, "
                f"Align: {train_losses['align']:.4f}, "
                f"KD: {train_losses['kd']:.4f}"
            )
            
            # 验证
            val_losses = self.validate()
            self.logger.info(
                f"Epoch {self.current_epoch} Val - "
                f"Total: {val_losses['total']:.4f}, "
                f"Traj: {val_losses['traj']:.4f}"
            )
            # 论文指标: L2 Error (m)
            self.logger.info(
                f"  L2 Error  - "
                f"@2.5s: {val_losses.get('l2_2.5s', -1):.4f}m, "
                f"@3.5s: {val_losses.get('l2_3.5s', -1):.4f}m, "
                f"@4.5s: {val_losses.get('l2_4.5s', -1):.4f}m, "
                f"Avg: {val_losses.get('l2_avg', -1):.4f}m"
            )
            # 论文指标: Collision Rate (%)
            self.logger.info(
                f"  Col Rate  - "
                f"@2.5s: {val_losses.get('col_2.5s', -1)*100:.2f}%, "
                f"@3.5s: {val_losses.get('col_3.5s', -1)*100:.2f}%, "
                f"@4.5s: {val_losses.get('col_4.5s', -1)*100:.2f}%, "
                f"Avg: {val_losses.get('col_avg', -1)*100:.2f}%"
            )
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning rate: {current_lr:.2e}")
            
            # 仅保存最优模型 (以 l2_avg 为准)
            l2_avg = val_losses.get('l2_avg', float('inf'))
            is_best = l2_avg < self.best_val_loss
            if is_best:
                self.best_val_loss = l2_avg
                self.save_checkpoint(val_losses=val_losses)
        
        self.logger.info(f"Training completed. Best L2 Avg: {self.best_val_loss:.4f}m")
    
    def save_checkpoint(self, val_losses: Dict = None):
        """仅保存最优模型检查点"""
        best_path = self.checkpoint_dir / "best_model.pt"
        self.model.save_checkpoint(
            str(best_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            val_losses=val_losses,
            config=self.config
        )
        self.logger.info(f"Best model saved to {best_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train V2X-VLM")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/autodl-tmp/outputs',
        help='Output directory (建议使用大容量数据盘)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置副本
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # 创建训练器
    trainer = Trainer(
        config=config,
        output_dir=str(output_dir)
    )
    
    # 创建数据加载器
    trainer.create_dataloaders()
    
    # 恢复训练
    if args.resume:
        trainer.model.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
