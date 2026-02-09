"""
测试模型前向传播是否正常工作

在运行训练前，先用这个脚本验证模型的基本功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

def test_model():
    """测试模型基本功能"""
    print("=" * 60)
    print("V2X-VLM 模型测试")
    print("=" * 60)
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 测试加载模型
    print("\n[1] 加载模型...")
    try:
        from src.models.v2x_vlm import V2XVLM
        
        model = V2XVLM(
            student_model_name="microsoft/Florence-2-base",
            teacher_model_name="microsoft/Florence-2-large",
            trajectory_length=45,
            hidden_dim=768,
            teacher_hidden_dim=1024,
            use_knowledge_distillation=True,
            use_contrastive_alignment=True,
            device=device
        )
        model = model.to(device)
        model.eval()
        print("✓ 模型加载成功")
        
        # 参数统计
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  可训练参数: {trainable:,} / {total:,}")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 测试 processor
    print("\n[2] 测试 Processor...")
    try:
        processor = model.get_processor()
        
        # 创建测试图像
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 448, 3), dtype=np.uint8)
        )
        text_prompt = "Scene: A driving scene. Task: Predict trajectory."
        
        inputs = processor(
            text=text_prompt,
            images=dummy_image,
            return_tensors="pt"
        )
        
        print(f"✓ Processor 工作正常")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")
        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"  attention_mask shape: {inputs['attention_mask'].shape}")
        
    except Exception as e:
        print(f"✗ Processor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试前向传播
    print("\n[3] 测试前向传播...")
    try:
        with torch.no_grad():
            pixel_values = inputs['pixel_values'].to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # 创建 dummy GT
            trajectory_gt = torch.randn(1, 45, 2).to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                trajectory_gt=trajectory_gt
            )
        
        print(f"✓ 前向传播成功")
        print(f"  trajectory_pred shape: {outputs['trajectory_pred'].shape}")
        
        if 'losses' in outputs:
            for loss_name, loss_val in outputs['losses'].items():
                if isinstance(loss_val, torch.Tensor):
                    print(f"  {loss_name}: {loss_val.item():.4f}")
                else:
                    print(f"  {loss_name}: {loss_val}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试训练模式（带梯度）
    print("\n[4] 测试训练模式...")
    try:
        model.train()
        
        pixel_values = inputs['pixel_values'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        trajectory_gt = torch.randn(1, 45, 2).to(device)
        
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            trajectory_gt=trajectory_gt
        )
        
        # 计算总损失
        total_loss = outputs['losses'].get('loss_traj', 0)
        if 'loss_align' in outputs['losses']:
            total_loss = total_loss + 0.1 * outputs['losses']['loss_align']
        if 'loss_kd' in outputs['losses']:
            total_loss = total_loss + 0.5 * outputs['losses']['loss_kd']
        
        # 反向传播测试
        total_loss.backward()
        
        # 检查梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"✓ 训练模式测试成功")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  有梯度的参数数量: {grad_count}")
        
    except Exception as e:
        print(f"✗ 训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试数据集
    print("\n[5] 测试数据集...")
    try:
        from src.data.dataset import V2XVLMDataset, V2XVLMCollator
        
        # 数据在上级目录
        data_root = Path(__file__).parent.parent.parent / "data"
        
        if data_root.exists():
            dataset = V2XVLMDataset(
                data_root=str(data_root),
                processor=processor,
                trajectory_horizon=45,
                image_size=(768, 768),
                split="train"
            )
            
            print(f"✓ 数据集加载成功")
            print(f"  样本数量: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  样本keys: {list(sample.keys())}")
                if 'pixel_values' in sample:
                    print(f"  pixel_values shape: {sample['pixel_values'].shape}")
                if 'trajectory_gt' in sample:
                    print(f"  trajectory_gt shape: {sample['trajectory_gt'].shape}")
        else:
            print(f"⚠ 数据目录不存在: {data_root}")
            
    except Exception as e:
        print(f"⚠ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_model()
