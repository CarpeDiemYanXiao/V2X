# V2X-VLM GT轨迹生成 - 正确方法与原始脚本错误分析

## 1. 概述

本文档分析DAIR-V2X数据集中GT (Ground Truth) 轨迹的**正确生成逻辑**，并详细说明原始 `v2x_vlm/scripts/make_gt_traj.py` 脚本中的**关键错误**。

---

## 2. DAIR-V2X 数据集结构

```
cooperative-vehicle-infrastructure/
├── cooperative/                    # ❌ 原始脚本错误使用
│   ├── data_info.json             # 协同配对信息
│   └── label/                     # 检测到的目标物体标签 (其他车辆/行人)
│
├── vehicle-side/                  # ✅ 正确数据来源
│   ├── data_info.json             # 车端帧信息
│   ├── image/                     # 车端相机图像
│   └── calib/
│       ├── novatel_to_world/      # ✅ ego车辆位姿 (GPS/IMU)
│       ├── lidar_to_novatel/
│       ├── lidar_to_camera/
│       └── camera_intrinsic/
│
└── infrastructure-side/           # 路端数据
    ├── data_info.json
    └── image/
```

---

## 3. 核心概念区分

### 3.1 Ego车辆位置 vs 检测目标位置

| 概念 | 来源 | 内容 | 用途 |
|------|------|------|------|
| **Ego位置** | `novatel_to_world/*.json` | 自车GPS/IMU位姿 | 轨迹预测GT |
| **检测目标** | `cooperative/label/*.json` | 场景中其他车辆、行人 | 障碍物检测 |

**关键区别：**
- `novatel_to_world` 包含的是**采集车辆本身**的位置（由GPS/IMU记录）
- `cooperative/label` 包含的是**场景中被检测到的其他物体**（车辆、行人等）

---

## 4. ✅ 正确的GT轨迹生成方法

### 4.1 数据来源

正确的轨迹来自 `vehicle-side/calib/novatel_to_world/` 目录下的位姿文件：

```json
// novatel_to_world/000000.json 示例
{
  "rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
  "translation": [[x], [y], [z]]
}
```

- `translation`: 车辆在世界坐标系中的 (x, y, z) 位置
- `rotation`: 3×3 旋转矩阵，可提取航向角 yaw

### 4.2 正确处理逻辑

```python
# ✅ 正确方法：使用 novatel_to_world 位姿数据
def generate_gt_trajectory(frame_id, batch_frames, horizon=45):
    """
    步骤：
    1. 按batch分组（同一连续驾驶场景）
    2. 按时间戳排序帧
    3. 从 novatel_to_world 读取位姿
    4. 收集未来 horizon 步的世界坐标
    5. 转换到ego中心坐标系
    """
    
    # 1. 读取当前帧位姿
    novatel_path = f"vehicle-side/calib/novatel_to_world/{frame_id}.json"
    current_pose = load_pose(novatel_path)  # (x, y, yaw)
    
    # 2. 收集未来帧位姿（世界坐标）
    future_poses = []
    for i, frame in enumerate(batch_frames[current_idx:current_idx+horizon]):
        pose = load_pose(frame['calib_novatel_to_world_path'])
        future_poses.append((pose['x'], pose['y']))
    
    # 3. 转换到ego中心坐标系
    ego_trajectory = world_to_ego_centric(future_poses, current_pose)
    
    return ego_trajectory  # shape: [45, 2]
```

### 4.3 坐标转换

```python
def world_to_ego_centric(trajectory, current_pose):
    """
    世界坐标 -> 自车中心坐标系
    
    Args:
        trajectory: [(wx, wy), ...] 世界坐标序列
        current_pose: (cx, cy, cyaw) 当前帧位姿
    
    Returns:
        ego_trajectory: [N, 2] 以当前位置为原点的相对坐标
    """
    cx, cy, cyaw = current_pose
    ego_trajectory = []
    
    for wx, wy in trajectory:
        # 1. 平移：减去当前位置
        dx = wx - cx
        dy = wy - cy
        
        # 2. 旋转：使自车朝向为x轴正方向
        cos_yaw = np.cos(-cyaw)
        sin_yaw = np.sin(-cyaw)
        ex = dx * cos_yaw - dy * sin_yaw
        ey = dx * sin_yaw + dy * cos_yaw
        
        ego_trajectory.append([ex, ey])
    
    return np.array(ego_trajectory)
```

---

## 5. ❌ 原始脚本错误分析

### 5.1 原始脚本代码 (`v2x_vlm/scripts/make_gt_traj.py`)

```python
# ❌ 错误代码
info_path = '.../cooperative/data_info.json'  # 错误：使用cooperative目录

for rec in tqdm(info, desc='building traj'):
    lbl_path = os.path.join(..., rec['cooperative_label_path'])  # ❌ 使用检测标签
    
    if os.path.exists(lbl_path):
        frame_objs = json.load(open(lbl_path))   # List[dict]
        # ❌ 致命错误：取标签中的第一辆 car 作为 ego
        ego = next((o for o in frame_objs if o.get('type') == 'car'), None)
        if ego is not None:
            last_xy = [ego['3d_location']['x'], ego['3d_location']['y']]  # ❌ 错误位置
```

### 5.2 核心错误列表

| 错误编号 | 错误描述 | 影响 |
|----------|----------|------|
| **E1** | 使用 `cooperative/label/` 作为位置来源 | 获取的是**检测目标**位置，不是ego车辆位置 |
| **E2** | 用 `type == 'car'` 筛选"ego" | 这些是**场景中其他车辆**，根本不是自己 |
| **E3** | 提取 `3d_location` 作为轨迹点 | 这是其他车辆的检测框位置 |
| **E4** | 没有按batch_id分组 | 混淆了不同驾驶场景的帧 |
| **E5** | 没有做坐标系转换 | 世界坐标直接当ego坐标用 |
| **E6** | 全局排序所有帧 | 连续性假设错误，不同场景被混在一起 |

### 5.3 错误示意图

```
原始脚本的理解 (❌ 错误):
┌─────────────────────────────────────┐
│  cooperative/label/*.json           │
│  ┌─────────────────────────────────┐│
│  │ [                               ││
│  │   {"type": "car", ...},  ← 错误地认为这是ego  │
│  │   {"type": "pedestrian", ...},  ││
│  │   {"type": "car", ...}          ││
│  │ ]                               ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘

实际数据含义 (✅ 正确):
┌─────────────────────────────────────┐
│  cooperative/label/*.json           │
│  ┌─────────────────────────────────┐│
│  │ [                               ││
│  │   {"type": "car", ...},  ← 这是场景中的其他车辆A ││
│  │   {"type": "pedestrian", ...},  ← 这是场景中的行人   ││
│  │   {"type": "car", ...}   ← 这是场景中的其他车辆B ││
│  │ ]                               ││
│  └─────────────────────────────────┘│
│                                     │
│  Ego车辆位置在哪里?                  │
│  → vehicle-side/calib/novatel_to_world/*.json │
└─────────────────────────────────────┘
```

### 5.4 错误后果

1. **轨迹语义完全错误**：生成的轨迹是**场景中某个随机车辆**的移动轨迹，而不是自车的未来规划轨迹

2. **轨迹不连续**：由于每帧选择的"第一辆car"可能是不同的车辆，导致轨迹在空间上跳跃

3. **训练目标混乱**：模型学习到的是预测"检测到的某辆车会去哪里"，而不是"自己应该如何行驶"

4. **性能极差**：L2误差可能极大，因为监督信号本身就是错误的

---

## 6. 数据验证

### 6.1 验证正确性

```python
# 验证生成的GT轨迹
import numpy as np

# 加载一个GT文件
traj = np.load("ground_truth_trajectories/000000.npy")

# 检查格式
assert traj.shape == (45, 2), f"形状错误: {traj.shape}"

# 检查合理性
# - 第一个点应该接近原点 (0, 0)，因为是ego中心坐标
print(f"起始点: {traj[0]}")  # 应该接近 [0, 0]

# - 轨迹应该是连续的，相邻点距离不应过大
distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
print(f"平均步长: {distances.mean():.3f}m")  # 正常值: 0.5~2m (取决于车速)
print(f"最大步长: {distances.max():.3f}m")   # 应该 < 5m
```

### 6.2 对比两种方法

| 指标 | 正确方法 | 原始脚本 |
|------|----------|----------|
| 数据来源 | novatel_to_world | cooperative/label |
| 坐标含义 | ego车GPS位置 | 检测目标位置 |
| 轨迹连续性 | ✅ 连续 | ❌ 跳跃 |
| 起始点 | 接近(0,0) | 随机位置 |
| 物理合理性 | ✅ 符合车辆运动学 | ❌ 可能有瞬移 |

---

## 7. 结论

### 原始脚本失败的根本原因

**原始脚本将"场景中检测到的其他车辆"的位置错误地当作"自车的未来轨迹"来使用。**

这是一个**概念性理解错误**：
- `cooperative/label/*.json` 是**目标检测**的结果，包含场景中被检测到的物体
- 这些物体**不包括采集车辆本身**（自己无法检测自己）
- **自车位置**记录在 `novatel_to_world/*.json` 中（来自GPS/IMU传感器）

### 正确做法

1. 使用 `vehicle-side/calib/novatel_to_world/*.json` 获取ego车辆位姿
2. 按 `batch_id` 分组帧，确保时序连续性
3. 按时间戳排序同一batch内的帧
4. 收集未来45帧的位姿，转换为ego中心坐标系
5. 保存为 `[45, 2]` 格式的numpy数组

---

## 8. 相关文件

| 文件 | 描述 |
|------|------|
| `v2x_vlm_reproduce/scripts/generate_trajectory_gt.py` | ✅ 正确实现 |
| `v2x_vlm/scripts/make_gt_traj.py` | ❌ 错误实现 |
| `data/.../vehicle-side/calib/novatel_to_world/` | ✅ 正确数据来源 |
| `data/.../cooperative/label/` | ❌ 错误数据来源 |
