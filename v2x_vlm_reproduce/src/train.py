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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import V2XVLMDataset, V2XVLMCollator, create_dataloaders
from models.v2x_vlm import V2XVLM
from losses.v2x_loss import V2XVLMLoss


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
        
        # 自动检测设备: NPU > CUDA > CPU
        if device == "auto":
            self.device = self._detect_device()
        else:
            self.device = device
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logging(str(self.output_dir))
        self.logger.info(f"Config: {config}")
        self.logger.info(f"Device: {self.device}")
        
        # 初始化模型
        self._init_model()
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 混合精度训练 (仅 CUDA 支持)
        self.use_amp = config.get('training', {}).get('use_amp', True) and 'cuda' in self.device
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
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
                return "npu"
        except ImportError:
            pass
        
        # 尝试 CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
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
        
        # 获取可训练参数
        params = self.model.get_trainable_parameters()
        
        # AdamW优化器
        self.optimizer = AdamW(
            params,
            lr=float(train_config.get('learning_rate', 1e-6)),
            weight_decay=float(train_config.get('weight_decay', 0.01)),
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler_type = train_config.get('scheduler', 'linear')
        epochs = train_config.get('epochs', 10)
        
        if scheduler_type == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=epochs
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-8
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: AdamW, lr={train_config.get('learning_rate', 1e-6)}")
        self.logger.info(f"Scheduler: {scheduler_type}")
    
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
                with autocast('cuda'):
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
        """验证"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'traj': 0.0,
            'l2_error': 0.0
        }
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
            
            # L2 Error计算
            trajectory_pred = outputs['trajectory_pred']
            l2_error = torch.sqrt(
                ((trajectory_pred - trajectory_gt) ** 2).sum(dim=-1)
            ).mean()
            
            val_losses['total'] += losses['total'].item()
            val_losses['traj'] += losses.get('loss_traj', torch.tensor(0.0)).item()
            val_losses['l2_error'] += l2_error.item()
            num_batches += 1
        
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
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
                f"Traj: {val_losses['traj']:.4f}, "
                f"L2 Error: {val_losses['l2_error']:.4f}m"
            )
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning rate: {current_lr:.2e}")
            
            # 保存检查点
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            self.save_checkpoint(is_best=is_best, val_losses=val_losses)
        
        self.logger.info(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False, val_losses: Dict = None):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pt"
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            val_losses=val_losses,
            config=self.config
        )
        
        if is_best:
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
        default='./outputs',
        help='Output directory'
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
