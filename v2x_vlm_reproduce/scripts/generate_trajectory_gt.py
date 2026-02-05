"""
GT轨迹生成脚本

从 novatel_to_world 位姿数据生成自车轨迹Ground Truth

轨迹格式: [45, 2] -> 45个时间步的 (x, y) 坐标 (自车中心坐标系)
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse


def load_pose(pose_path):
    """
    加载单帧位姿
    
    Args:
        pose_path: novatel_to_world/*.json 路径
    
    Returns:
        x, y, yaw: 世界坐标位置和航向角
    """
    with open(pose_path, 'r') as f:
        data = json.load(f)
    
    # 提取平移 (世界坐标)
    x = data['translation'][0][0]
    y = data['translation'][1][0]
    
    # 从旋转矩阵提取航向角
    rotation = np.array(data['rotation'])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    
    return x, y, yaw


def world_to_ego_centric(trajectory, current_pose):
    """
    将世界坐标轨迹转换为自车中心坐标系
    
    论文描述: ego vehicle's position in the Virtual World Coordinate System
    转换到: ego-centric coordinate system (当前帧为原点, 自车朝向为x轴正方向)
    
    Args:
        trajectory: [(x, y), ...] 世界坐标序列
        current_pose: (x, y, yaw) 当前帧位姿
    
    Returns:
        ego_trajectory: [N, 2] 以当前帧为原点的相对坐标
    """
    cx, cy, cyaw = current_pose
    ego_trajectory = []
    
    for wx, wy in trajectory:
        # 1. 平移到自车中心
        dx = wx - cx
        dy = wy - cy
        
        # 2. 旋转到自车朝向 (使自车朝向为x轴正方向)
        cos_yaw = np.cos(-cyaw)
        sin_yaw = np.sin(-cyaw)
        ex = dx * cos_yaw - dy * sin_yaw
        ey = dx * sin_yaw + dy * cos_yaw
        
        ego_trajectory.append([ex, ey])
    
    return np.array(ego_trajectory, dtype=np.float32)


def load_vehicle_data_info(data_root):
    """
    加载车端元数据
    
    Returns:
        data_info: 帧信息列表
        frame_to_meta: 帧ID到元数据的映射
    """
    vehicle_side = Path(data_root) / "cooperative-vehicle-infrastructure/vehicle-side"
    data_info_path = vehicle_side / "data_info.json"
    
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # 构建帧ID到元数据的映射
    frame_to_meta = {}
    for item in data_info:
        frame_id = item['image_path'].split('/')[-1].replace('.jpg', '')
        frame_to_meta[frame_id] = item
    
    return data_info, frame_to_meta


def group_frames_by_batch(data_info):
    """
    按batch_id分组帧 (同一连续驾驶场景)
    
    Returns:
        batches: {batch_id: [帧列表按时间排序]}
    """
    batches = defaultdict(list)
    
    for item in data_info:
        batch_id = item.get('batch_id', 'unknown')
        batches[batch_id].append(item)
    
    # 对每个batch按时间戳排序
    for batch_id in batches:
        batches[batch_id].sort(key=lambda x: int(x['image_timestamp']))
    
    return batches


def generate_trajectories(
    data_root,
    output_dir,
    horizon=45,
    fps=10,
    verbose=True
):
    """
    生成所有帧的GT轨迹
    
    Args:
        data_root: 数据集根目录
        output_dir: 输出目录
        horizon: 预测时间步数 (45 = 4.5秒)
        fps: 帧率 (10Hz)
        verbose: 是否打印详细信息
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    novatel_dir = data_root / "cooperative-vehicle-infrastructure/vehicle-side/calib/novatel_to_world"
    
    # 加载元数据
    data_info, frame_to_meta = load_vehicle_data_info(data_root)
    
    # 按batch分组
    batches = group_frames_by_batch(data_info)
    
    if verbose:
        print(f"数据集统计:")
        print(f"  - 总帧数: {len(data_info)}")
        print(f"  - Batch数量: {len(batches)}")
        print(f"  - 位姿文件目录: {novatel_dir}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 预测horizon: {horizon} 帧 ({horizon/fps:.1f} 秒)")
    
    valid_count = 0
    skip_count = 0
    skip_reasons = defaultdict(int)
    
    for batch_id, frames in tqdm(batches.items(), desc="Processing batches"):
        # 提取帧ID列表
        frame_ids = [f['image_path'].split('/')[-1].replace('.jpg', '') for f in frames]
        
        for i, frame in enumerate(frames):
            frame_id = frame_ids[i]
            
            # 检查是否有足够的未来帧 (需要i+1到i+horizon共horizon个未来帧)
            if i + horizon >= len(frames):
                skip_count += 1
                skip_reasons['insufficient_future_frames'] += 1
                continue
            
            # 加载当前帧位姿
            current_pose_path = novatel_dir / f"{frame_id}.json"
            if not current_pose_path.exists():
                skip_count += 1
                skip_reasons['missing_current_pose'] += 1
                continue
            
            try:
                current_pose = load_pose(current_pose_path)
            except Exception as e:
                skip_count += 1
                skip_reasons['pose_load_error'] += 1
                continue
            
            # 加载未来轨迹
            future_trajectory = []
            valid = True
            
            for j in range(1, horizon + 1):
                future_frame_id = frame_ids[i + j]
                future_pose_path = novatel_dir / f"{future_frame_id}.json"
                
                if not future_pose_path.exists():
                    valid = False
                    break
                
                try:
                    fx, fy, _ = load_pose(future_pose_path)
                    future_trajectory.append([fx, fy])
                except Exception:
                    valid = False
                    break
            
            if not valid or len(future_trajectory) != horizon:
                skip_count += 1
                skip_reasons['incomplete_future_trajectory'] += 1
                continue
            
            # 转换为自车中心坐标系
            ego_trajectory = world_to_ego_centric(future_trajectory, current_pose)
            
            # 验证轨迹形状
            assert ego_trajectory.shape == (horizon, 2), \
                f"轨迹形状错误: {ego_trajectory.shape}, 应为 ({horizon}, 2)"
            
            # 保存
            output_path = output_dir / f"{frame_id}.npy"
            np.save(output_path, ego_trajectory)
            valid_count += 1
    
    if verbose:
        print(f"\n生成完成:")
        print(f"  - 有效轨迹: {valid_count}")
        print(f"  - 跳过: {skip_count}")
        if skip_reasons:
            print(f"  - 跳过原因:")
            for reason, count in skip_reasons.items():
                print(f"      {reason}: {count}")
    
    return valid_count


def verify_trajectories(trajectory_dir, num_samples=5):
    """
    验证生成的轨迹
    """
    trajectory_dir = Path(trajectory_dir)
    files = list(trajectory_dir.glob("*.npy"))
    
    if not files:
        print("❌ 未找到轨迹文件!")
        return False
    
    print(f"\n轨迹验证:")
    print(f"  - 总文件数: {len(files)}")
    
    # 抽样检查
    samples = files[:num_samples]
    print(f"\n  抽样检查 ({len(samples)} 个):")
    
    all_valid = True
    for f in samples:
        traj = np.load(f)
        shape_ok = traj.shape == (45, 2)
        dtype_ok = traj.dtype == np.float32
        range_ok = np.abs(traj).max() < 1000  # 合理范围检查
        
        status = "✅" if (shape_ok and dtype_ok and range_ok) else "❌"
        print(f"    {f.name}: shape={traj.shape}, dtype={traj.dtype}, "
              f"range=[{traj.min():.2f}, {traj.max():.2f}] {status}")
        
        if not (shape_ok and dtype_ok):
            all_valid = False
    
    # 统计
    shapes = []
    for f in tqdm(files, desc="  统计中"):
        traj = np.load(f)
        shapes.append(traj.shape)
    
    unique_shapes = set(shapes)
    print(f"\n  形状统计: {unique_shapes}")
    
    if len(unique_shapes) == 1 and (45, 2) in unique_shapes:
        print("  ✅ 所有轨迹形状正确!")
    else:
        print("  ❌ 存在形状不一致的轨迹!")
        all_valid = False
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='生成GT轨迹')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                        default='../data/ground_truth_trajectories',
                        help='输出目录')
    parser.add_argument('--horizon', type=int, default=45,
                        help='预测时间步数')
    parser.add_argument('--fps', type=int, default=10,
                        help='帧率')
    parser.add_argument('--verify', action='store_true',
                        help='生成后验证')
    args = parser.parse_args()
    
    print("=" * 60)
    print("V2X-VLM GT轨迹生成器")
    print("=" * 60)
    
    # 生成轨迹
    valid_count = generate_trajectories(
        data_root=args.data_root,
        output_dir=args.output_dir,
        horizon=args.horizon,
        fps=args.fps
    )
    
    # 验证
    if args.verify:
        verify_trajectories(args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"完成! 生成 {valid_count} 个GT轨迹")
    print("=" * 60)


if __name__ == "__main__":
    main()
