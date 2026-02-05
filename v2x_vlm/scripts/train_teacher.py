# scripts/train_teacher.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

# 添加项目根目录到路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# 确保能找到 Florence-2 的路径 (根据你的环境调整)
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
        
        # 让 processor 处理截断，这里不手动截断
        # prompt = prompt[:512] 
        
        # 确保尺寸一致
        if v_img.size != i_img.size:
            i_img = i_img.resize(v_img.size)
        
        # 水平拼接图像 [Iv, Ii]
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
        format='%(asctime)s - [TeacherTrain] - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    # 1. 加载配置
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    logger = setup_logger()
    
    # 修改输出目录
    original_output_dir = cfg.output_dir
    cfg.output_dir = os.path.join(original_output_dir, "teacher_training")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"训练 Teacher 模型 | 设备: {device} | 输出: {cfg.output_dir}")

    # [核心] 初始化 Tokenizer
    tokenizer = TrajectoryTokenizer(cfg)

    # 2. 数据准备
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

    # 3. 初始化 Teacher 模型
    logger.info("初始化 Teacher 模型 (Florence-2-large)...")
    teacher = V2XVLMTeacher(cfg).to(device)

    # 4. 配置参数冻结策略
    # 可以通过环境变量或配置选择训练策略
    freeze_backbone = os.getenv("FREEZE_BACKBONE", "false").lower() == "true"
    
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
                # 微调模式：解冻backbone，但用小学习率
                param.requires_grad = True
                backbone_params.append(param)
    
    if freeze_backbone:
        logger.info("🔥 策略: 完全冻结Backbone | 只训练预测头")
    else:
        logger.info("🔥 策略: 视觉冻结 | Backbone微调(2e-5) | Head快训(1e-3)")

    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    logger.info(f"Teacher 可训练参数量: {trainable_params/1e6:.2f}M")
    # 预期：参数量应该大幅减少，只剩几百万或者几千万

    # 调整学习率：根据训练策略选择
    if freeze_backbone:
        # 完全冻结模式：只优化预测头，使用较大学习率
        optimizer = torch.optim.AdamW(
            head_params,
            lr=1e-3,
            weight_decay=0.01
        )
        # 简单的cosine调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=1e-5
        )
    else:
        # 微调模式：分别设置backbone和head的学习率
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5},
            {'params': head_params, 'lr': 1e-3}
        ], weight_decay=0.01)
        
        # 使用warmup + cosine调度器
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

    # [核心] 梯度累积步数（只在微调模式下使用）
    accumulation_steps = 8 if not freeze_backbone else 1
    
    best_val_l2 = float('inf')

    # 6. 训练循环
    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        teacher.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            combined_images = batch["combined_images"]
            prompts = batch["prompts"]
            
            gt_traj_coords = batch["trajectory"].to(device)
            gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)
            
            # 【关键修改】必须加上这行！把 0 (Padding/无效位) 变成 -100，防止模型偷懒学 0
            gt_tokens[gt_tokens == 0] = -100 
            gt_tokens[gt_tokens == 1023] = -100

            outputs = teacher(combined_images, prompts)
            traj_logits = outputs["trajectory_logits"]
            
            # 计算分类 Loss，使用较小的label smoothing（0.05）提高训练稳定性
            loss = trajectory_loss(traj_logits, gt_tokens, label_smoothing=0.05)

            # [核心] 梯度累积
            loss = loss / accumulation_steps
            loss.backward()

            # 梯度累积（只在微调模式下）
            if freeze_backbone:
                # 完全冻结模式：每个batch都更新
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()
            else:
                # 微调模式：使用梯度累积
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            if freeze_backbone:
                epoch_loss += loss.item()
            else:
                epoch_loss += loss.item() * accumulation_steps
            
            # Debug 打印 (只在每个 epoch 的第一个 batch 打印)
            if batch_idx == 0:
                last_step_idx = 44 
                reshaped_logits = traj_logits.view(-1, 45, 2, 1024)
                pred_probs = torch.softmax(reshaped_logits[0, last_step_idx, 0], dim=-1)
                max_prob, max_idx = torch.max(pred_probs, dim=-1)
                
                # 对应的真值
                gt_val = gt_tokens[0, last_step_idx, 0].item()
                
                # 计算top-5预测
                top5_probs, top5_indices = torch.topk(pred_probs, k=5)
                
                # 修正打印逻辑：如果真值是 -100，说明这步是无效位
                gt_display = "PADDING" if gt_val == -100 else gt_val
                # 显示两个参数组的学习率
                backbone_lr = optimizer.param_groups[0]['lr']
                head_lr = optimizer.param_groups[1]['lr']
                # 获取当前调度器步数
                current_scheduler_step = scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else 0
                print(f"\n[Debug T=4.5s X-axis] GT Token: {gt_display} | Pred Max: {max_idx.item()} (Prob: {max_prob.item():.4f})")
                print(f"[Debug LR] Backbone: {backbone_lr:.2e} | Head: {head_lr:.2e} | Scheduler Step: {current_scheduler_step}")
                print(f"[Debug Top-5] {[(idx.item(), prob.item()) for idx, prob in zip(top5_indices, top5_probs)]}")
        avg_loss = epoch_loss / len(train_loader)
        if freeze_backbone:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            scheduler.step()  # 在epoch结束时更新
        else:
            backbone_lr = optimizer.param_groups[0]['lr']
            head_lr = optimizer.param_groups[1]['lr']
            logger.info(f"Train Loss: {avg_loss:.4f} | LR Backbone: {backbone_lr:.2e} | LR Head: {head_lr:.2e}")

        # 验证 (监控 L2 误差)
        val_l2 = validate(teacher, val_loader, device, tokenizer)
        logger.info(f"Val L2 Error: {val_l2:.4f} meters")

        # 保存最佳模型 (根据 L2 误差)
        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            save_path = os.path.join(cfg.output_dir, "teacher_best.pth")
            torch.save(teacher.state_dict(), save_path)
            logger.info(f"保存最佳模型至: {save_path}")

@torch.no_grad()
def validate(model, loader, device, tokenizer):
    model.eval()
    total_l2_error = 0.0
    total_valid_points = 0  # 统计有效点数
    
    pbar = tqdm(loader, desc="Validating")
    
    for batch in pbar:
        combined_images = batch["combined_images"]
        prompts = batch["prompts"]
        
        gt_traj_coords = batch["trajectory"].to(device)
        gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)
        # 验证集也要处理 Padding，以便生成 Mask
        gt_tokens[gt_tokens == 0] = -100

        outputs = model(combined_images, prompts)
        traj_logits = outputs["trajectory_logits"]

        # 1. 预测 Token
        pred_token_ids = torch.argmax(traj_logits, dim=-1).view(-1, 45, 2)
        
        # 2. 转回坐标
        pred_coords = tokenizer.tokens_to_coords(pred_token_ids)
        
        # 3. 创建 Mask：只计算不是 -100 的点
        # 只要 (x, y) 中有一个是 -100，这个点就是无效的
        mask = (gt_tokens != -100).all(dim=-1) # [B, 45]
        
        # 4. 计算距离
        distances = torch.norm(pred_coords - gt_traj_coords, dim=-1)
        valid_distances = distances[mask] # 只取有效距离
        
        if valid_distances.numel() > 0:
            total_l2_error += valid_distances.sum().item()
            total_valid_points += valid_distances.numel()
            
            pbar.set_postfix({"L2": f"{valid_distances.mean().item():.2f}m"})
        
    avg_l2 = total_l2_error / total_valid_points if total_valid_points > 0 else 0.0
    return avg_l2



if __name__ == "__main__":
    main()