# scripts/train_teacher_regression.py
"""
直接回归坐标的训练方案
不使用分类，直接预测坐标值
可能比分类方法更简单有效
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-base")
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-large")

from src.utils.config import load_config
from src.data.dataset import V2XVLMDataset
from src.models.v2x_vlm_teacher_regression import V2XVLMTeacherRegression

def collate_fn(batch):
    """拼接车辆-基础设施图像"""
    vehicle_imgs = [b["vehicle_img"] for b in batch]
    infra_imgs = [b["infra_img"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    trajectories = torch.stack([b["trajectory"] for b in batch])

    combined_images = []
    for v_img, i_img, prompt in zip(vehicle_imgs, infra_imgs, prompts):
        if not prompt.strip():
            raise ValueError(f"Empty prompt found: {prompt}")
        
        if v_img.size != i_img.size:
            i_img = i_img.resize(v_img.size)
        
        combined_width = v_img.width + i_img.width
        combined_img = Image.new('RGB', (combined_width, v_img.height))
        combined_img.paste(v_img, (0, 0))
        combined_img.paste(i_img, (v_img.width, 0))
        
        combined_images.append(combined_img)

    return {
        "combined_images": combined_images,
        "prompts": prompts,
        "trajectory": trajectories
    }

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [RegressionTrain] - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    logger = setup_logger()
    
    cfg.output_dir = os.path.join(cfg.output_dir, "teacher_training_regression")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"训练 Teacher 模型（直接回归）| 设备: {device} | 输出: {cfg.output_dir}")

    # 数据准备
    train_dataset = V2XVLMDataset(split="train", cfg=cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = V2XVLMDataset(split="val", cfg=cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 初始化模型
    logger.info("初始化 Teacher 模型 (Florence-2-large, Regression)...")
    teacher = V2XVLMTeacherRegression(cfg).to(device)

    # 配置参数冻结策略：默认完全冻结backbone（更稳定）
    freeze_backbone = os.getenv("FREEZE_BACKBONE", "true").lower() == "true"
    
    head_params = []
    backbone_params = []
    
    for name, param in teacher.named_parameters():
        # A. 视觉部分：永远冻结
        if "vision" in name.lower() or "davit" in name.lower():
            param.requires_grad = False
            
        # B. 预测头 (Head)：必须训练
        elif "traj_head" in name or "feature_fusion" in name:
            param.requires_grad = True
            head_params.append(param)
            
        # C. 语言模型 (Backbone)：根据策略决定
        else:
            if freeze_backbone:
                # 完全冻结模式：只训练预测头
                param.requires_grad = False
            else:
                # 部分微调模式：解冻backbone，但用小学习率
                param.requires_grad = True
                backbone_params.append(param)
    
    if freeze_backbone:
        logger.info("🔥 策略: 完全冻结Backbone | 只训练回归头")
    else:
        logger.info("🔥 策略: 视觉冻结 | Backbone微调(5e-5) | Head快训(1e-2)")

    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"可训练参数: {trainable_params/1e6:.2f}M / 总参数: {total_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # 优化器：根据训练策略选择
    if freeze_backbone:
        # 完全冻结模式：只优化预测头，使用更大的学习率让模型快速学习
        optimizer = torch.optim.AdamW(
            head_params,
            lr=1e-2,  # 提高学习率，让预测范围快速扩大
            weight_decay=0.01
        )
        # 使用更激进的学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=1e-4  # 提高最小学习率
        )
        accumulation_steps = 1  # 不需要梯度累积
    else:
        # 部分微调模式：分别设置backbone和head的学习率
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},  # Backbone学习率提高
            {'params': head_params, 'lr': 1e-2}       # Head学习率提高
        ], weight_decay=0.01)
        
        # 使用warmup + cosine调度器
        accumulation_steps = 4  # 梯度累积，减少显存占用
        num_actual_updates_per_epoch = len(train_loader) // accumulation_steps
        num_warmup_updates = min(10, num_actual_updates_per_epoch // 10)
        num_training_updates = num_actual_updates_per_epoch * cfg.epochs
        
        def lr_lambda(step):
            if step < num_warmup_updates:
                warmup_start = 0.1
                warmup_end = 1.0
                if num_warmup_updates > 0:
                    return warmup_start + (warmup_end - warmup_start) * (step + 1) / num_warmup_updates
                else:
                    return 1.0
            else:
                progress = (step - num_warmup_updates) / max(1, (num_training_updates - num_warmup_updates))
                return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))
        
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        logger.info(f"学习率调度: Warmup {num_warmup_updates}步, 总更新 {num_training_updates}步, 梯度累积 {accumulation_steps}步")
    
    best_val_l2 = float('inf')

    # 训练循环
    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        teacher.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            combined_images = batch["combined_images"]
            prompts = batch["prompts"]
            
            gt_traj_coords = batch["trajectory"].to(device)  # [B, 45, 2]

            outputs = teacher(combined_images, prompts)
            pred_coords = outputs["trajectory_coords"]  # [B, 45, 2]
            
            # 使用Huber Loss（对异常值鲁棒，结合了L1和L2的优点）
            # delta=1.0: 当误差<1米时用L2，误差>1米时用L1
            loss = nn.functional.huber_loss(pred_coords, gt_traj_coords, reduction='mean', delta=10.0)
            
            # 根据训练策略选择梯度累积方式
            if freeze_backbone:
                # 完全冻结模式：每个batch都更新
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            else:
                # 部分微调模式：使用梯度累积
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()  # 更新学习率
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps
            
            # Debug打印：增加更多信息
            if batch_idx == 0:
                last_step_idx = 44
                pred_val = pred_coords[0, last_step_idx].detach().cpu().numpy()
                gt_val = gt_traj_coords[0, last_step_idx].cpu().numpy()
                error = np.linalg.norm(pred_val - gt_val)
                current_lr = optimizer.param_groups[0]['lr']
                
                # 打印整条轨迹的统计信息
                pred_all = pred_coords[0].detach().cpu().numpy()
                gt_all = gt_traj_coords[0].cpu().numpy()
                pred_range = f"[{pred_all.min():.1f}, {pred_all.max():.1f}]"
                gt_range = f"[{gt_all.min():.1f}, {gt_all.max():.1f}]"
                
                print(f"\n[Debug T=4.5s] GT: {gt_val} | Pred: {pred_val} | Error: {error:.2f}m | LR: {current_lr:.2e}")
                print(f"[Range] GT: {gt_range} | Pred: {pred_range} | Loss: {loss.item():.4f}")
        
        if freeze_backbone:
            scheduler.step()  # 在epoch结束时更新
            avg_loss = epoch_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        else:
            avg_loss = epoch_loss / len(train_loader)
            backbone_lr = optimizer.param_groups[0]['lr']
            head_lr = optimizer.param_groups[1]['lr']
            logger.info(f"Train Loss: {avg_loss:.4f} | LR Backbone: {backbone_lr:.2e} | LR Head: {head_lr:.2e}")

        # 验证
        val_l2 = validate(teacher, val_loader, device)
        logger.info(f"Val L2 Error: {val_l2:.4f} meters")

        # 保存最佳模型
        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            save_path = os.path.join(cfg.output_dir, "teacher_best_regression.pth")
            torch.save(teacher.state_dict(), save_path)
            logger.info(f"保存最佳模型至: {save_path}")

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_l2_error = 0.0
    total_valid_points = 0
    
    pbar = tqdm(loader, desc="Validating")
    
    for batch in pbar:
        combined_images = batch["combined_images"]
        prompts = batch["prompts"]
        
        gt_traj_coords = batch["trajectory"].to(device)

        outputs = model(combined_images, prompts)
        pred_coords = outputs["trajectory_coords"]
        
        # 计算L2距离
        distances = torch.norm(pred_coords - gt_traj_coords, dim=-1)  # [B, 45]
        
        total_l2_error += distances.sum().item()
        total_valid_points += distances.numel()
        
        pbar.set_postfix({"L2": f"{distances.mean().item():.2f}m"})
        
    avg_l2 = total_l2_error / total_valid_points if total_valid_points > 0 else 0.0
    return avg_l2

if __name__ == "__main__":
    main()
