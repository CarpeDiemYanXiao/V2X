# scripts/train_teacher_frozen.py
"""
完全冻结Backbone的训练方案
只训练预测头，将Florence-2作为特征提取器
优点：训练稳定、快速、不容易过拟合
缺点：可能性能上限较低
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

# 添加项目根目录到路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-base")
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-large")

from src.utils.config import load_config
from src.data.dataset import V2XVLMDataset
from src.models.v2x_vlm_teacher import V2XVLMTeacher
from src.models.losses import trajectory_loss
from src.utils.tokenizer import TrajectoryTokenizer

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
        format='%(asctime)s - [FrozenTrain] - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    logger = setup_logger()
    
    cfg.output_dir = os.path.join(cfg.output_dir, "teacher_training_frozen")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"训练 Teacher 模型（完全冻结Backbone）| 设备: {device} | 输出: {cfg.output_dir}")

    tokenizer = TrajectoryTokenizer(cfg)

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
    logger.info("初始化 Teacher 模型 (Florence-2-large)...")
    teacher = V2XVLMTeacher(cfg).to(device)

    # 【关键】完全冻结Backbone，只训练预测头
    logger.info("🔥 策略: 完全冻结Backbone | 只训练预测头")
    head_params = []
    
    for name, param in teacher.named_parameters():
        # 只训练预测头相关的参数
        if "traj_head" in name or "feature_fusion" in name:
            param.requires_grad = True
            head_params.append(param)
        else:
            # 冻结所有其他参数（包括vision、language model等）
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"可训练参数: {trainable_params/1e6:.2f}M / 总参数: {total_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # 只优化预测头，使用较大的学习率
    optimizer = torch.optim.AdamW(
        head_params,
        lr=1e-3,  # 只训练头，可以用较大学习率
        weight_decay=0.01
    )

    # 简单的学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-5
    )
    
    best_val_l2 = float('inf')

    # 训练循环
    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        teacher.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            combined_images = batch["combined_images"]
            prompts = batch["prompts"]
            
            gt_traj_coords = batch["trajectory"].to(device)
            gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)
            
            # 处理padding
            gt_tokens[gt_tokens == 0] = -100 
            gt_tokens[gt_tokens == 1023] = -100

            outputs = teacher(combined_images, prompts)
            traj_logits = outputs["trajectory_logits"]
            
            loss = trajectory_loss(traj_logits, gt_tokens, label_smoothing=0.05)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Debug打印
            if batch_idx == 0:
                last_step_idx = 44 
                reshaped_logits = traj_logits.view(-1, 45, 2, 1024)
                pred_probs = torch.softmax(reshaped_logits[0, last_step_idx, 0], dim=-1)
                max_prob, max_idx = torch.max(pred_probs, dim=-1)
                
                gt_val = gt_tokens[0, last_step_idx, 0].item()
                gt_display = "PADDING" if gt_val == -100 else gt_val
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n[Debug T=4.5s X-axis] GT Token: {gt_display} | Pred Max: {max_idx.item()} (Prob: {max_prob.item():.4f}) | LR: {current_lr:.2e}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # 验证
        val_l2 = validate(teacher, val_loader, device, tokenizer)
        logger.info(f"Val L2 Error: {val_l2:.4f} meters")

        # 保存最佳模型
        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            save_path = os.path.join(cfg.output_dir, "teacher_best_frozen.pth")
            torch.save(teacher.state_dict(), save_path)
            logger.info(f"保存最佳模型至: {save_path}")

@torch.no_grad()
def validate(model, loader, device, tokenizer):
    model.eval()
    total_l2_error = 0.0
    total_valid_points = 0
    
    pbar = tqdm(loader, desc="Validating")
    
    for batch in pbar:
        combined_images = batch["combined_images"]
        prompts = batch["prompts"]
        
        gt_traj_coords = batch["trajectory"].to(device)
        gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)
        gt_tokens[gt_tokens == 0] = -100

        outputs = model(combined_images, prompts)
        traj_logits = outputs["trajectory_logits"]

        pred_token_ids = torch.argmax(traj_logits, dim=-1).view(-1, 45, 2)
        pred_coords = tokenizer.tokens_to_coords(pred_token_ids)
        
        mask = (gt_tokens != -100).all(dim=-1)
        distances = torch.norm(pred_coords - gt_traj_coords, dim=-1)
        valid_distances = distances[mask]
        
        if valid_distances.numel() > 0:
            total_l2_error += valid_distances.sum().item()
            total_valid_points += valid_distances.numel()
            pbar.set_postfix({"L2": f"{valid_distances.mean().item():.2f}m"})
        
    avg_l2 = total_l2_error / total_valid_points if total_valid_points > 0 else 0.0
    return avg_l2

if __name__ == "__main__":
    main()
