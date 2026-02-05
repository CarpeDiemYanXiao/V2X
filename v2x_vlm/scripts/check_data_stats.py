"""
检查数据集的统计信息，帮助诊断训练问题
"""
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)
    
    data_root = Path(cfg.data_root)
    
    # 读取训练集ID
    split_file = data_root / "split" / "train.txt"
    with open(split_file) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    print(f"训练集样本数: {len(train_ids)}")
    print("\n正在分析轨迹数据...")
    
    all_coords = []
    all_distances = []
    
    for frame_id in train_ids[:1000]:  # 只检查前1000个样本
        traj_path = data_root / "ground_truth_trajectories" / f"{frame_id}.npy"
        if not traj_path.exists():
            continue
            
        traj = np.load(traj_path)  # [45, 2]
        
        # 应用与dataset.py相同的转换
        current_pos = traj[0].copy()
        traj_trans = traj - current_pos
        
        dx = traj_trans[1, 0] - traj_trans[0, 0]
        dy = traj_trans[1, 1] - traj_trans[0, 1]
        
        if abs(dx) < 1e-5 and abs(dy) < 1e-5:
            theta = 0.0
        else:
            theta = np.arctan2(dy, dx)
        
        c = np.cos(-theta)
        s = np.sin(-theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        traj_rotated = np.matmul(traj_trans, rot_mat.T)
        
        # 收集统计信息
        all_coords.append(traj_rotated.flatten())
        
        # 计算每个时间步的距离
        distances = np.linalg.norm(traj_rotated, axis=1)
        all_distances.append(distances[-1])  # 最后一个点的距离
    
    all_coords = np.concatenate(all_coords)
    all_distances = np.array(all_distances)
    
    print("\n" + "="*60)
    print("坐标统计 (转换后的自车坐标系)")
    print("="*60)
    print(f"最小值: {all_coords.min():.2f} 米")
    print(f"最大值: {all_coords.max():.2f} 米")
    print(f"平均值: {all_coords.mean():.2f} 米")
    print(f"标准差: {all_coords.std():.2f} 米")
    print(f"中位数: {np.median(all_coords):.2f} 米")
    print(f"25%分位: {np.percentile(all_coords, 25):.2f} 米")
    print(f"75%分位: {np.percentile(all_coords, 75):.2f} 米")
    
    print("\n" + "="*60)
    print("终点距离统计 (4.5秒后)")
    print("="*60)
    print(f"最小距离: {all_distances.min():.2f} 米")
    print(f"最大距离: {all_distances.max():.2f} 米")
    print(f"平均距离: {all_distances.mean():.2f} 米")
    print(f"标准差: {all_distances.std():.2f} 米")
    print(f"中位数: {np.median(all_distances):.2f} 米")
    
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    
    # 根据数据范围给出建议
    coord_range = all_coords.max() - all_coords.min()
    if coord_range > 200:
        print("⚠️  坐标范围很大 (>200米)，建议:")
        print("   1. 检查数据预处理是否正确")
        print("   2. 考虑对坐标进行归一化")
        print("   3. 使用Huber Loss代替MSE Loss")
    
    if all_distances.mean() > 50:
        print("⚠️  平均终点距离较大 (>50米)，这是正常的")
        print("   - 4.5秒后车辆可能行驶较远")
        print("   - 模型需要学习长期预测")
    
    if all_coords.std() > 30:
        print("⚠️  坐标标准差较大 (>30米)，建议:")
        print("   1. 对坐标进行标准化: (x - mean) / std")
        print("   2. 在训练时使用更大的学习率")
        print("   3. 增加模型容量")

if __name__ == "__main__":
    main()
