# scripts/diagnose_training.py
"""
训练诊断脚本
检查数据质量、特征质量、模型输出等
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-base")
sys.path.append("/root/autodl-tmp/models/florence2/Florence-2-large")

from src.utils.config import load_config
from src.data.dataset import V2XVLMDataset
from src.models.v2x_vlm_teacher_regression import V2XVLMTeacherRegression

def collate_fn(batch):
    vehicle_imgs = [b["vehicle_img"] for b in batch]
    infra_imgs = [b["infra_img"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    trajectories = torch.stack([b["trajectory"] for b in batch])

    from PIL import Image
    combined_images = []
    for v_img, i_img in zip(vehicle_imgs, infra_imgs):
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

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # 1. 检查数据分布
    print("=" * 60)
    print("1. 检查数据分布")
    print("=" * 60)
    
    train_dataset = V2XVLMDataset(split="train", cfg=cfg)
    all_coords = []
    all_coords_abs = []
    
    for i in range(min(100, len(train_dataset))):  # 检查前100个样本
        data = train_dataset[i]
        traj = data["trajectory"].numpy()
        all_coords.append(traj)
        all_coords_abs.append(np.abs(traj))
    
    all_coords = np.concatenate(all_coords, axis=0)
    all_coords_abs = np.concatenate(all_coords_abs, axis=0)
    
    print(f"坐标范围: X=[{all_coords[:, 0].min():.2f}, {all_coords[:, 0].max():.2f}], "
          f"Y=[{all_coords[:, 1].min():.2f}, {all_coords[:, 1].max():.2f}]")
    print(f"坐标绝对值范围: X=[{all_coords_abs[:, 0].min():.2f}, {all_coords_abs[:, 0].max():.2f}], "
          f"Y=[{all_coords_abs[:, 1].min():.2f}, {all_coords_abs[:, 1].max():.2f}]")
    print(f"坐标均值: X={all_coords[:, 0].mean():.2f}, Y={all_coords[:, 1].mean():.2f}")
    print(f"坐标标准差: X={all_coords[:, 0].std():.2f}, Y={all_coords[:, 1].std():.2f}")
    
    # 检查异常值
    threshold = 500
    abnormal_x = np.abs(all_coords[:, 0]) > threshold
    abnormal_y = np.abs(all_coords[:, 1]) > threshold
    if abnormal_x.any() or abnormal_y.any():
        print(f"⚠️  发现异常值: X方向{abnormal_x.sum()}个, Y方向{abnormal_y.sum()}个")
    else:
        print("✅ 未发现异常值")
    
    # 2. 检查特征质量
    print("\n" + "=" * 60)
    print("2. 检查特征质量")
    print("=" * 60)
    
    model = V2XVLMTeacherRegression(cfg).to(device)
    model.eval()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    all_features = []
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="提取特征", total=min(10, len(train_loader)))):
            if batch_idx >= 10:
                break
                
            combined_images = batch["combined_images"]
            prompts = batch["prompts"]
            gt_coords = batch["trajectory"].to(device)
            
            # 获取中间特征
            inputs = model.processor(
                images=combined_images,
                text=prompts,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            batch_size = inputs["pixel_values"].shape[0]
            decoder_start_token_id = model.model.config.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = model.model.config.bos_token_id
            
            decoder_input_ids = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=device
            )
            
            outputs = model.model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 提取特征
            if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
                encoder_hidden = outputs.encoder_last_hidden_state
                encoder_pooled = encoder_hidden.mean(dim=1)
            else:
                encoder_pooled = None
            
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                decoder_hidden = outputs.last_hidden_state
                decoder_pooled = decoder_hidden[:, -1, :]
            else:
                decoder_pooled = None
            
            if encoder_pooled is not None:
                all_features.append(encoder_pooled.cpu().numpy())
            if decoder_pooled is not None:
                all_features.append(decoder_pooled.cpu().numpy())
            
            # 获取预测
            pred_outputs = model(combined_images, prompts)
            pred_coords = pred_outputs["trajectory_coords"]
            
            all_preds.append(pred_coords.cpu().numpy())
            all_gts.append(gt_coords.cpu().numpy())
    
    if all_features:
        all_features = np.concatenate(all_features, axis=0)
        print(f"特征形状: {all_features.shape}")
        print(f"特征均值: {all_features.mean():.6f}")
        print(f"特征标准差: {all_features.std():.6f}")
        print(f"特征范围: [{all_features.min():.6f}, {all_features.max():.6f}]")
        
        # 检查特征是否过于相似（说明特征质量不好）
        if len(all_features) > 1:
            # 计算特征之间的相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(all_features[:10])
            avg_similarity = similarity[np.triu_indices(10, k=1)].mean()
            print(f"特征平均相似度: {avg_similarity:.4f} (越接近1说明特征越相似，可能质量不好)")
    
    # 3. 检查预测质量
    print("\n" + "=" * 60)
    print("3. 检查预测质量")
    print("=" * 60)
    
    if all_preds and all_gts:
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
        
        errors = np.linalg.norm(all_preds - all_gts, axis=-1)
        print(f"预测误差统计:")
        print(f"  均值: {errors.mean():.2f}米")
        print(f"  中位数: {np.median(errors):.2f}米")
        print(f"  标准差: {errors.std():.2f}米")
        print(f"  最小值: {errors.min():.2f}米")
        print(f"  最大值: {errors.max():.2f}米")
        
        print(f"\n预测值范围: X=[{all_preds[:, :, 0].min():.2f}, {all_preds[:, :, 0].max():.2f}], "
              f"Y=[{all_preds[:, :, 1].min():.2f}, {all_preds[:, :, 1].max():.2f}]")
        print(f"真实值范围: X=[{all_gts[:, :, 0].min():.2f}, {all_gts[:, :, 0].max():.2f}], "
              f"Y=[{all_gts[:, :, 1].min():.2f}, {all_gts[:, :, 1].max():.2f}]")
        
        # 检查预测是否集中在某个值
        pred_x_mean = all_preds[:, :, 0].mean()
        pred_y_mean = all_preds[:, :, 1].mean()
        print(f"\n预测值均值: X={pred_x_mean:.2f}, Y={pred_y_mean:.2f}")
        print(f"真实值均值: X={all_gts[:, :, 0].mean():.2f}, Y={all_gts[:, :, 1].mean():.2f}")
        
        if abs(pred_x_mean) < 5 and abs(pred_y_mean) < 5:
            print("⚠️  预测值集中在0附近，说明模型可能没有学到有效的映射")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
