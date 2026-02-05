# scripts/train.py
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from PIL import Image
import time

# 添加项目根目录到路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-base")
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-large")

from src.utils.config import load_config
from src.data.dataset import V2XVLMDataset
from src.models.v2x_vlm_student import V2XVLMStudent
from src.models.v2x_vlm_teacher import V2XVLMTeacher
from src.models.losses import trajectory_loss, contrastive_loss, kd_loss
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
        
        prompt = prompt[:512]
        
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
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    logger = setup_logger()
    
    # 防止覆盖逻辑
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_output_dir = cfg.output_dir
    cfg.output_dir = os.path.join(original_output_dir, f"train_{timestamp}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    logger.info(f"输出目录: {cfg.output_dir}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # [Step 4 新增] 初始化 Tokenizer
    tokenizer = TrajectoryTokenizer(cfg)

    logger.info("初始化数据集...")
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

    logger.info("初始化模型...")
    student = V2XVLMStudent(cfg).to(device)
    teacher = V2XVLMTeacher(cfg).to(device)
    
    # 加载训练好的 Teacher 权重
    teacher_weight_path = os.path.join(original_output_dir, "teacher_training", "teacher_best.pth")
    
    if os.path.exists(teacher_weight_path):
        logger.info(f"加载 Teacher 权重: {teacher_weight_path}")
        teacher.load_state_dict(torch.load(teacher_weight_path, map_location=device))
    else:
        logger.warning(f"⚠️ 未找到 Teacher 权重: {teacher_weight_path}，Teacher 将输出随机噪声！(请先运行 train_teacher.py)")

    # 冻结 Teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    student.train()
    
    # 优化器配置
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 0.01))
    )

    best_val_loss = float('inf')
    logger.info("开始训练...")

    for epoch in range(cfg.epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{cfg.epochs} =====")
        student.train()
        
        epoch_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_align_loss = 0.0
        epoch_kd_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            combined_images = batch["combined_images"]
            prompts = batch["prompts"]
            
            # [Step 4 修改] 获取 GT 坐标并转为 Token
            gt_traj_coords = batch["trajectory"].to(device)
            gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)

            # 学生模型前向
            student_outputs = student(combined_images, prompts)
            # 注意：现在 student_outputs["trajectory_logits"] 是 [B, 90, 1024]
            student_logits = student_outputs["trajectory_logits"]
            
            visual_feat = student_outputs["visual_features"]
            text_feat = student_outputs["text_features"]

            # 教师模型前向（冻结）
            with torch.no_grad():
                teacher_outputs = teacher(combined_images, prompts)
                # Teacher 也输出 [B, 90, 1024]
                teacher_logits = teacher_outputs["trajectory_logits"]

            # 计算损失
            # 1. 轨迹生成损失 (CrossEntropy)
            loss_traj = trajectory_loss(student_logits, gt_tokens)
            
            # 2. 对比损失
            loss_align = contrastive_loss(visual_feat, text_feat, temperature=0.07)
            
            # 3. 知识蒸馏损失 (KL Divergence)
            loss_kd = kd_loss(student_logits, teacher_logits, temperature=2.0)
            
            total_loss = (
                loss_traj + 
                float(cfg.lambda_align) * loss_align + 
                float(cfg.lambda_kd) * loss_kd
            )

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            max_norm = getattr(cfg, "grad_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)
            optimizer.step()

            # 累计损失
            epoch_loss += total_loss.item()
            epoch_traj_loss += loss_traj.item()
            epoch_align_loss += loss_align.item()
            epoch_kd_loss += loss_kd.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"  Batch [{batch_idx}/{len(train_loader)}]: "
                    f"total={total_loss.item():.4f}, "
                    f"traj={loss_traj.item():.4f}, "
                    f"align={loss_align.item():.4f}, "
                    f"kd={loss_kd.item():.4f}"
                )

        # 统计
        avg_total_loss = epoch_loss / len(train_loader)
        avg_traj_loss = epoch_traj_loss / len(train_loader)
        avg_align_loss = epoch_align_loss / len(train_loader)
        avg_kd_loss = epoch_kd_loss / len(train_loader)
        
        logger.info(
            f"[Epoch {epoch+1}] Avg Loss: total={avg_total_loss:.4f}, "
            f"traj={avg_traj_loss:.4f}, align={avg_align_loss:.4f}, kd={avg_kd_loss:.4f}"
        )

        # 验证
        val_loss = validate(student, val_loader, device, tokenizer, logger)
        
        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(cfg.output_dir, "best_model.pth")
            torch.save(student.state_dict(), best_model_path)
            logger.info(f"保存最佳模型: {best_model_path}")

        # 定期保存
        if (epoch + 1) % getattr(cfg, "save_interval", 2) == 0:
            checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg
            }, checkpoint_path)

@torch.no_grad()
def validate(student, val_loader, device, tokenizer, logger):
    student.eval()
    total_loss = 0.0
    total_l2_error = 0.0  # 新增
    total_samples = 0

    for batch in tqdm(val_loader, desc="Validation"):
        combined_images = batch["combined_images"]
        prompts = batch["prompts"]
        
        # 1. 准备 GT
        gt_traj_coords = batch["trajectory"].to(device) # [B, 45, 2]
        gt_tokens = tokenizer.coords_to_tokens(gt_traj_coords).to(device)

        # 2. 前向
        student_outputs = student(combined_images, prompts)
        student_logits = student_outputs["trajectory_logits"] # [B, 90, 1024]

        # 3. 计算分类 Loss
        loss_traj = trajectory_loss(student_logits, gt_tokens)
        total_loss += loss_traj.item() * gt_tokens.shape[0]
        
        # 4. [新增] 计算 L2 Error (米) 用于监控性能
        # 取概率最大的 Token ID
        pred_token_ids = torch.argmax(student_logits, dim=-1) # [B, 90]
        # Reshape 回 [B, 45, 2]
        B = pred_token_ids.shape[0]
        pred_token_ids = pred_token_ids.view(B, -1, 2)
        # 转回坐标
        pred_coords = tokenizer.tokens_to_coords(pred_token_ids) # [B, 45, 2]
        # 计算与 GT 的欧氏距离 (L2)
        l2_error = torch.norm(pred_coords - gt_traj_coords, dim=-1).mean() # 平均误差
        total_l2_error += l2_error.item() * B

        total_samples += gt_tokens.shape[0]

    avg_loss = total_loss / total_samples
    avg_l2 = total_l2_error / total_samples # 新增
    
    logger.info(f"[Validation] CE Loss: {avg_loss:.4f} | Avg L2 Error: {avg_l2:.4f} meters")
    student.train()
    return avg_loss # 或者返回 avg_l2 作为保存模型的依据

if __name__ == "__main__":
    main()