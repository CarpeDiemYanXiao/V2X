"""
数据验证脚本

验证数据完整性和格式正确性
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def verify_data_integrity(data_root, verbose=True):
    """
    验证数据完整性
    
    检查项目:
    1. 图像目录存在且包含文件
    2. GT轨迹目录存在且格式正确
    3. 元数据文件存在
    4. 位姿文件存在
    5. 配对数据正确
    """
    data_root = Path(data_root)
    issues = []
    stats = {}
    
    print("=" * 60)
    print("V2X-VLM 数据验证")
    print("=" * 60)
    
    # 1. 检查车端图像
    v_img_dir = data_root / "cooperative-vehicle-infrastructure-vehicle-side-image"
    if not v_img_dir.exists():
        issues.append(f"❌ 车端图像目录不存在: {v_img_dir}")
    else:
        v_count = len(list(v_img_dir.glob("*.jpg")))
        stats['vehicle_images'] = v_count
        print(f"✅ 车端图像: {v_count} 张")
    
    # 2. 检查路侧图像
    i_img_dir = data_root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
    if not i_img_dir.exists():
        issues.append(f"❌ 路侧图像目录不存在: {i_img_dir}")
    else:
        i_count = len(list(i_img_dir.glob("*.jpg")))
        stats['infra_images'] = i_count
        print(f"✅ 路侧图像: {i_count} 张")
    
    # 3. 检查GT轨迹
    traj_dir = data_root / "ground_truth_trajectories"
    if not traj_dir.exists():
        issues.append(f"❌ GT轨迹目录不存在: {traj_dir}")
        issues.append("   请运行: python scripts/generate_trajectory_gt.py")
    else:
        traj_files = list(traj_dir.glob("*.npy"))
        traj_count = len(traj_files)
        stats['trajectories'] = traj_count
        
        if traj_count == 0:
            issues.append(f"❌ GT轨迹目录为空!")
        else:
            print(f"✅ GT轨迹: {traj_count} 个")
            
            # 验证轨迹格式
            sample = traj_files[0]
            arr = np.load(sample)
            if arr.shape == (45, 2):
                print(f"   ✅ 轨迹格式正确: {arr.shape}")
            else:
                issues.append(f"❌ 轨迹格式错误: {arr.shape}, 应为 (45, 2)")
    
    # 4. 检查配对数据
    coop_info = data_root / "cooperative-vehicle-infrastructure/cooperative/data_info.json"
    if coop_info.exists():
        with open(coop_info, 'r') as f:
            coop_data = json.load(f)
        stats['cooperative_pairs'] = len(coop_data)
        print(f"✅ 配对帧数: {len(coop_data)}")
    else:
        issues.append(f"❌ 配对数据不存在: {coop_info}")
    
    # 5. 检查车端元数据
    vehicle_info = data_root / "cooperative-vehicle-infrastructure/vehicle-side/data_info.json"
    if vehicle_info.exists():
        with open(vehicle_info, 'r') as f:
            vehicle_data = json.load(f)
        stats['vehicle_frames'] = len(vehicle_data)
        print(f"✅ 车端帧数: {len(vehicle_data)}")
        
        # 统计batch
        batches = set(item.get('batch_id', 'unknown') for item in vehicle_data)
        stats['batches'] = len(batches)
        print(f"   ✅ Batch数量: {len(batches)}")
    else:
        issues.append(f"❌ 车端元数据不存在: {vehicle_info}")
    
    # 6. 检查位姿数据
    pose_dir = data_root / "cooperative-vehicle-infrastructure/vehicle-side/calib/novatel_to_world"
    if pose_dir.exists():
        pose_count = len(list(pose_dir.glob("*.json")))
        stats['pose_files'] = pose_count
        print(f"✅ 位姿文件: {pose_count} 个")
    else:
        issues.append(f"❌ 位姿目录不存在: {pose_dir}")
    
    # 7. 检查场景描述 (可选)
    scene_desc = data_root / "scene_descriptions.json"
    if scene_desc.exists():
        with open(scene_desc, 'r') as f:
            desc_data = json.load(f)
        v_desc = len(desc_data.get('vehicle', {}))
        i_desc = len(desc_data.get('infrastructure', {}))
        stats['scene_descriptions_vehicle'] = v_desc
        stats['scene_descriptions_infra'] = i_desc
        print(f"✅ 场景描述: 车端 {v_desc}, 路侧 {i_desc}")
    else:
        print(f"⚠️  场景描述文件不存在 (可选): {scene_desc}")
        print(f"   可运行: python scripts/generate_scene_descriptions.py")
    
    # 汇总
    print("\n" + "=" * 60)
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n请解决以上问题后再开始训练")
    else:
        print("✅ 数据验证通过!")
    print("=" * 60)
    
    # 统计概览
    if verbose and stats:
        print("\n数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return len(issues) == 0, stats


def verify_sample_matching(data_root, num_samples=10):
    """
    验证样本匹配关系
    
    检查配对数据中的图像是否都存在
    """
    data_root = Path(data_root)
    
    print("\n" + "=" * 60)
    print("验证样本匹配关系")
    print("=" * 60)
    
    # 加载配对数据
    coop_info = data_root / "cooperative-vehicle-infrastructure/cooperative/data_info.json"
    with open(coop_info, 'r') as f:
        coop_data = json.load(f)
    
    v_img_dir = data_root / "cooperative-vehicle-infrastructure-vehicle-side-image"
    i_img_dir = data_root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
    traj_dir = data_root / "ground_truth_trajectories"
    
    valid_count = 0
    missing_vehicle = 0
    missing_infra = 0
    missing_traj = 0
    
    samples = coop_data[:num_samples] if num_samples else coop_data
    
    for item in samples:
        v_frame = item['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
        i_frame = item['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')
        
        v_exists = (v_img_dir / f"{v_frame}.jpg").exists()
        i_exists = (i_img_dir / f"{i_frame}.jpg").exists()
        t_exists = (traj_dir / f"{v_frame}.npy").exists() if traj_dir.exists() else False
        
        if v_exists and i_exists and t_exists:
            valid_count += 1
        else:
            if not v_exists:
                missing_vehicle += 1
            if not i_exists:
                missing_infra += 1
            if not t_exists:
                missing_traj += 1
    
    total = len(samples)
    print(f"检查样本数: {total}")
    print(f"  ✅ 有效样本: {valid_count} ({valid_count/total*100:.1f}%)")
    if missing_vehicle:
        print(f"  ❌ 缺失车端图像: {missing_vehicle}")
    if missing_infra:
        print(f"  ❌ 缺失路侧图像: {missing_infra}")
    if missing_traj:
        print(f"  ⚠️  缺失GT轨迹: {missing_traj}")
    
    return valid_count == total


def main():
    parser = argparse.ArgumentParser(description='验证数据')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='数据集根目录')
    parser.add_argument('--check_samples', type=int, default=100,
                        help='检查的样本数量')
    args = parser.parse_args()
    
    # 验证数据完整性
    success, stats = verify_data_integrity(args.data_root)
    
    # 验证样本匹配
    if success:
        verify_sample_matching(args.data_root, args.check_samples)


if __name__ == "__main__":
    main()
