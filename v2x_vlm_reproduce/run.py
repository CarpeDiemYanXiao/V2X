#!/usr/bin/env python
"""
V2X-VLM 快速启动脚本

支持设备: CUDA / NPU (华为昇腾) / CPU

一键执行完整复现流程:
1. 数据预处理 (生成GT轨迹)
2. 生成场景描述
3. 验证数据完整性
4. 训练模型
5. 评估模型
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """运行命令并检查结果"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)
    
    print(f"\n✅ Completed: {description}")


def main():
    parser = argparse.ArgumentParser(description="V2X-VLM 快速启动")
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='数据集根目录路径 (包含cooperative-vehicle-infrastructure/)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'npu', 'cpu'],
        help='训练设备: auto(自动检测) | cuda | npu | cpu'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='使用指定的配置文件 (优先级高于自动生成)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='输出目录'
    )
    parser.add_argument(
        '--skip_preprocess',
        action='store_true',
        help='跳过数据预处理'
    )
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='跳过训练,仅评估'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='用于评估的检查点路径'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小'
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print("\n" + "="*60)
    print("V2X-VLM Reproduction Pipeline")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print("="*60)
    
    python_exe = sys.executable
    
    # Step 1: 生成GT轨迹
    if not args.skip_preprocess:
        run_command(
            [python_exe, 'scripts/generate_trajectory_gt.py',
             '--data_root', str(data_root),
             '--output_dir', str(data_root / 'ground_truth_trajectories')],
            "生成GT轨迹"
        )
        
        # Step 2: 生成场景描述 (可选,耗时较长)
        print("\n" + "-"*60)
        print("跳过场景描述生成 (可选步骤,需要GPU)")
        print("如需生成,请手动运行: python scripts/generate_scene_descriptions.py")
        print("-"*60)
        
        # Step 3: 验证数据
        run_command(
            [python_exe, 'scripts/verify_data.py',
             '--data_root', str(data_root)],
            "验证数据完整性"
        )
    
    # 创建临时配置文件
    config_path = output_dir / 'temp_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据设备设置 num_workers
    if args.device == 'npu' or args.device == 'cpu':
        num_workers = 0
        pin_memory = 'false'
    else:
        num_workers = 4
        pin_memory = 'false'  # Windows 建议关闭
    
    config_content = f"""
# V2X-VLM 训练配置 (自动生成)

device:
  type: "{args.device}"

model:
  student_model: "microsoft/Florence-2-base"
  teacher_model: "microsoft/Florence-2-large"
  trajectory_length: 45
  hidden_dim: 768
  teacher_hidden_dim: 1024
  projection_dim: 256
  temperature: 0.07
  kd_temperature: 2.0
  use_kd: true
  use_contrastive: true

data:
  root: "{str(data_root).replace(chr(92), '/')}"
  image_size: [768, 768]
  trajectory_horizon: 45
  train_ratio: 0.8
  num_workers: {num_workers}
  pin_memory: {pin_memory}

training:
  batch_size: {args.batch_size}
  epochs: {args.epochs}
  learning_rate: 1.0e-6
  weight_decay: 0.01
  grad_clip: 1.0
  scheduler: "linear"
  use_amp: true

loss:
  lambda_align: 0.1
  lambda_kd: 0.5
  contrastive_temperature: 0.07
  kd_temperature: 2.0
  use_contrastive: true
  use_kd: true
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n配置文件已保存到: {config_path}")
    
    # Step 4: 训练
    if not args.skip_train:
        run_command(
            [python_exe, 'src/train.py',
             '--config', str(config_path),
             '--output_dir', str(output_dir)],
            "训练模型"
        )
    
    # Step 5: 评估
    checkpoint = args.checkpoint
    if checkpoint is None:
        # 查找最新的best_model.pt
        checkpoint_candidates = list(output_dir.glob('**/best_model.pt'))
        if checkpoint_candidates:
            checkpoint = str(sorted(checkpoint_candidates)[-1])
    
    if checkpoint and Path(checkpoint).exists():
        run_command(
            [python_exe, 'src/evaluate.py',
             '--config', str(config_path),
             '--checkpoint', checkpoint,
             '--output_dir', str(output_dir / 'eval_results'),
             '--visualize'],
            "评估模型"
        )
    else:
        print("\n⚠️ 未找到检查点文件,跳过评估")
    
    print("\n" + "="*60)
    print("✅ V2X-VLM 复现流程完成!")
    print("="*60)
    print(f"\n结果目录: {output_dir}")
    print("\n目标指标 (论文 Table 2):")
    print("  - L2 Error Avg: 1.21m")
    print("  - Collision Rate Avg: 0.03%")
    print("="*60)


if __name__ == "__main__":
    main()
