# V2X-VLM 论文复现

基于 "V2X-VLM: End-to-End V2X cooperative autonomous driving through large vision-Language models" 论文的完整复现实现。

**支持设备**: CUDA / NPU (华为昇腾) / CPU

## 目录结构

```
v2x_vlm_reproduce/
├── configs/
│   ├── train_config.yaml      # 通用训练配置
│   └── train_config_npu.yaml  # NPU云服务器配置
├── scripts/
│   ├── generate_trajectory_gt.py    # GT轨迹生成
│   ├── generate_scene_descriptions.py  # 场景描述生成
│   ├── test_model.py          # 模型测试脚本
│   └── verify_data.py         # 数据验证
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         # 数据集类
│   ├── models/
│   │   ├── __init__.py
│   │   ├── v2x_vlm.py         # 主模型 (Teacher-Student架构)
│   │   ├── trajectory_head.py # 轨迹预测头 (MLP)
│   │   └── feature_alignment.py # 对比特征对齐
│   ├── losses/
│   │   ├── __init__.py
│   │   └── v2x_loss.py        # 综合损失函数
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py         # 评估指标
│   │   └── visualization.py   # 可视化
│   ├── __init__.py
│   ├── train.py               # 训练脚本
│   └── evaluate.py            # 评估脚本
├── run.py                     # 一键运行脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
cd v2x_vlm_reproduce
pip install -r requirements.txt

# NPU环境额外安装 (华为昇腾)
# pip install torch_npu
```

### 2. 一键运行 (推荐)

```bash
# CUDA (自动检测)
python run.py --data_root ../data --epochs 10

# 指定 NPU
python run.py --data_root ../data --device npu --epochs 10

# 使用指定配置文件
python run.py --data_root ../data --config configs/train_config_npu.yaml
```

### 3. 手动分步执行

#### 3.1 数据预处理

```bash
# 生成GT轨迹 (最关键步骤!)
python scripts/generate_trajectory_gt.py --data_root ../data --verify

# 验证数据完整性
python scripts/verify_data.py --data_root ../data

# 生成场景描述 (可选,需要GPU,耗时较长)
python scripts/generate_scene_descriptions.py --data_root ../data
```

#### 3.2 训练

```bash
# 通用配置 (CUDA)
python src/train.py --config configs/train_config.yaml --output_dir outputs

# NPU 配置
python src/train.py --config configs/train_config_npu.yaml --output_dir outputs
```

#### 3.3 评估

```bash
python src/evaluate.py --config configs/train_config.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --visualize
```

## 设备支持

| 设备 | 配置文件 | 备注 |
|------|---------|------|
| CUDA | `train_config.yaml` | 推荐，支持混合精度 |
| NPU | `train_config_npu.yaml` | 华为昇腾，需安装 torch_npu |
| CPU | `train_config.yaml` + `device: cpu` | 不推荐，速度慢 |

## 论文核心方法

### 1. 模型架构 (Section 4)

- **Florence-2 VLM**: 使用预训练的视觉语言模型
  - Teacher: Florence-2-Large (冻结)
  - Student: Florence-2-Base (可训练)

- **图像拼接**: 车端 + 路侧图像沿宽度拼接
  ```
  [I_v, I_i] ∈ R^{H × (W_v + W_i) × 3}
  ```

- **轨迹解码器**: 简单MLP网络
  ```
  f_traj(F_multi) → τ = {(x_t, y_t) | t = 1, ..., 45}
  ```

### 2. 损失函数 (Eq.14)

```
L_total = L_traj + λ₁ × L_align + λ₂ × L_KD
```

- **L_traj** (Eq.8): 轨迹回归损失 (L1)
- **L_align** (Eq.12): 对比特征对齐 (InfoNCE, κ=0.07)
- **L_KD** (Eq.13): 知识蒸馏 (KL散度, T=2.0)

### 3. 训练配置 (Section 5)

| 参数 | 值 |
|------|------|
| Batch Size | 4 |
| Learning Rate | 1e-6 |
| Epochs | 10 |
| Optimizer | AdamW |
| λ_align | 0.1 |
| λ_kd | 0.5 |
| Temperature (KD) | 2.0 |

## 目标指标 (Table 2)

| 指标 | 2.5s | 3.5s | 4.5s | 平均 |
|------|------|------|------|------|
| L2 Error (m) ↓ | 1.09 | 1.12 | 1.42 | **1.21** |
| Collision Rate (%) ↓ | 0.02 | 0.03 | 0.03 | **0.03** |

## 常见问题

### Q: ground_truth_trajectories 目录为空?

这是训练效果差的主要原因! 运行以下命令生成GT轨迹:

```bash
python scripts/generate_trajectory_gt.py --data_root ../data --verify
```

### Q: GPU显存不足?

1. 减小batch_size到2或1
2. 使用混合精度训练 (默认开启)
3. 关闭知识蒸馏 (不加载Teacher模型)

### Q: 如何只使用Student模型?

修改 `configs/train_config.yaml`:
```yaml
model:
  use_kd: false
  use_contrastive: false
```

## 引用

```bibtex
@article{v2x-vlm,
  title={V2X-VLM: End-to-End V2X cooperative autonomous driving through large vision-Language models},
  author={...},
  journal={...},
  year={2024}
}
```
