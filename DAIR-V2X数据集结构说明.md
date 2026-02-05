# DAIR-V2X 数据集结构说明

## 概述

DAIR-V2X是一个车路协同自动驾驶数据集,包含车端(Vehicle-side)、路侧(Infrastructure-side)和协同(Cooperative)三部分数据。

## 目录结构

当前数据集采用**解压后独立存放**的结构：

```
data/
├── cooperative-vehicle-infrastructure/
│   ├── vehicle-side/          # 车端元数据
│   │   ├── calib/             # 标定文件
│   │   │   ├── novatel_to_world/     # ⭐ GPS位姿 -> 世界坐标系 (自车轨迹来源)
│   │   │   ├── lidar_to_novatel/     # LiDAR -> GPS坐标系
│   │   │   ├── lidar_to_camera/      # LiDAR -> 相机坐标系
│   │   │   └── camera_intrinsic/     # 相机内参
│   │   ├── label/             # 标注数据
│   │   │   ├── camera/        # 相机标注 (其他车辆检测框)
│   │   │   └── lidar/         # LiDAR标注
│   │   └── data_info.json     # 帧元数据索引
│   │
│   ├── infrastructure-side/   # 路侧元数据
│   │   ├── calib/             # 标定文件
│   │   │   ├── virtuallidar_to_world/  # 虚拟LiDAR -> 世界坐标系
│   │   │   ├── virtuallidar_to_camera/ # 虚拟LiDAR -> 相机
│   │   │   └── camera_intrinsic/       # 相机内参
│   │   ├── label/             # 标注数据
│   │   │   ├── virtuallidar/  # 虚拟LiDAR标注
│   │   │   └── camera/        # 相机标注
│   │   └── data_info.json     # 帧元数据索引
│   │
│   └── cooperative/           # 车路协同数据 (6,617 帧配对)
│       ├── label_world/       # 世界坐标系下的3D标注
│       └── data_info.json     # 车路帧配对关系
│
├── cooperative-vehicle-infrastructure-vehicle-side-image/      # ⭐ 车端图像 (.jpg)
│   └── *.jpg                  # 约 15,000+ 图像文件
│
└── cooperative-vehicle-infrastructure-infrastructure-side-image/  # ⭐ 路侧图像 (.jpg)
    └── *.jpg                  # 约 12,000+ 图像文件
```

> **注意**: V2X-VLM模型仅使用图像数据，不需要LiDAR点云 (`velodyne/`)。

## 核心数据文件格式

### 1. vehicle-side/data_info.json (车端元数据)

每一帧包含以下字段:

```json
{
  "image_path": "image/020502.jpg",
  "image_timestamp": "1626247179676000",        // 图像时间戳 (微秒)
  "pointcloud_path": "velodyne/020502.pcd",
  "pointcloud_timestamp": "1626247179628753",   // 点云时间戳 (微秒)
  "calib_novatel_to_world_path": "calib/novatel_to_world/020502.json",  // ⭐ 关键位姿
  "calib_lidar_to_novatel_path": "calib/lidar_to_novatel/020502.json",
  "calib_lidar_to_camera_path": "calib/lidar_to_camera/020502.json",
  "calib_camera_intrinsic_path": "calib/camera_intrinsic/020502.json",
  "label_camera_std_path": "label/camera/020502.json",
  "label_lidar_std_path": "label/lidar/020502.json",
  "batch_id": "34",           // ⭐ 序列ID (同一连续驾驶场景)
  "batch_start_id": "004015", // ⭐ 序列起始帧
  "batch_end_id": "004226",   // ⭐ 序列结束帧
  "intersection_loc": ""
}
```

**关键字段说明:**
- `batch_id`: 标识连续驾驶场景,用于构建轨迹序列
- `batch_start_id` / `batch_end_id`: 该序列的帧ID范围
- 帧率约为 10Hz (帧间隔约100ms)

### 2. calib/novatel_to_world/*.json (⭐ 自车世界位姿)

这是**自车轨迹GT的数据来源**:

```json
{
  "translation": [
    [2636.189362599922],    // X坐标 (米)
    [1745.0222184006125],   // Y坐标 (米)
    [21.4352444]            // Z坐标 (米, 高程)
  ],
  "rotation": [
    [-0.8603104583236002, 0.5097612830524799, 0.003107152796159999],
    [-0.50973699943776, -0.8601673754765603, -0.01675059238608],
    [-0.00586613177472, -0.01599453787344, 0.9998548711088]
  ]
}
```

- `translation`: 自车在世界坐标系下的位置 [x, y, z]
- `rotation`: 3×3旋转矩阵,表示自车朝向

### 3. cooperative/data_info.json (车路配对关系)

```json
{
  "infrastructure_image_path": "infrastructure-side/image/000084.jpg",
  "infrastructure_pointcloud_path": "infrastructure-side/velodyne/000084.pcd",
  "vehicle_image_path": "vehicle-side/image/015438.jpg",
  "vehicle_pointcloud_path": "vehicle-side/velodyne/015438.pcd",
  "cooperative_label_path": "cooperative/label_world/015438.json",
  "system_error_offset": {
    "delta_x": 0,   // 系统校准偏移
    "delta_y": 0
  }
}
```

### 4. cooperative/label_world/*.json (世界坐标3D检测标注)

包含周围车辆/物体的3D检测标注(非自车轨迹):

```json
[
  {
    "type": "truck",              // 类型: car/truck/van/bus
    "occluded_state": 1,          // 遮挡状态
    "truncated_state": 0,         // 截断状态
    "3d_location": {              // 世界坐标位置
      "x": 2688.5436,
      "y": 1687.9704,
      "z": 21.7195
    },
    "3d_dimensions": {            // 尺寸 (高/宽/长)
      "h": 3.857966,
      "w": 2.862032,
      "l": 6.797563
    },
    "rotation": 0.05151201,       // 航向角 (弧度)
    "2d_box": {...},              // 图像2D框
    "world_8_points": [...]       // 世界坐标8顶点
  }
]
```

### 5. infrastructure-side/data_info.json (路侧元数据)

```json
{
  "pointcloud_path": "velodyne/019937.pcd",
  "pointcloud_timestamp": "1626165671000291",
  "lidar_id": "151",
  "intersection_loc": "yizhuang09",            // 交叉口位置
  "calib_camera_intrinsic_path": "calib/camera_intrinsic/019937.json",
  "calib_virtuallidar_to_world_path": "calib/virtuallidar_to_world/019937.json",
  "calib_virtuallidar_to_camera_path": "calib/virtuallidar_to_camera/019937.json",
  "image_path": "image/019937.jpg",
  "image_timestamp": "1626165671102639",
  "camera_ip": "172_18_9_101",
  "camera_id": "1",
  "batch_id": "122",
  "valid_batch_splits": [
    {
      "batch_start_id": "019816",
      "batch_end_id": "019955"
    }
  ]
}
```

## V2X-VLM轨迹预测所需数据

### 模型输入

1. **车端图像**: `data/cooperative-vehicle-infrastructure-vehicle-side-image/*.jpg`
2. **路侧图像**: `data/cooperative-vehicle-infrastructure-infrastructure-side-image/*.jpg`
3. **配对关系**: `cooperative/data_info.json`

### Ground Truth (需要生成)

V2X-VLM论文需要**自车未来45帧轨迹** (4.5秒 @ 10Hz):

```
轨迹格式: [45, 2] -> 45个时间步的 (x, y) 坐标
```

**GT轨迹生成方法:**

从 `calib/novatel_to_world/*.json` 提取连续帧位置:

```python
# 伪代码
for frame_id in current_batch:
    pose = load_json(f"calib/novatel_to_world/{frame_id}.json")
    x = pose["translation"][0][0]
    y = pose["translation"][1][0]
    trajectory.append([x, y])

# 转换为自车中心坐标系 (以当前帧为原点)
ego_trajectory = world_to_ego_centric(trajectory, current_pose)

# 保存为 .npy
np.save(f"ground_truth_trajectories/{frame_id}.npy", ego_trajectory)
```

## 数据统计

| 数据类型 | 帧数 |
|---------|------|
| 车端帧 (vehicle-side) | 15,285 |
| 路侧帧 (infrastructure-side) | 12,424 |
| 配对帧 (cooperative) | 6,617 |
| 序列数 (batch数量) | ~150+ |

## 注意事项

1. **图像路径**: 图像文件存放在独立目录，需要调整dataset.py中的路径映射
   - 车端图像: `data/cooperative-vehicle-infrastructure-vehicle-side-image/`
   - 路侧图像: `data/cooperative-vehicle-infrastructure-infrastructure-side-image/`
2. **LiDAR数据**: V2X-VLM仅使用视觉数据，**不需要** `velodyne/` 点云文件
3. **坐标系统**: 数据使用世界坐标系 (WGS84 -> UTM投影)，单位为米
4. **时间同步**: 车路数据通过时间戳对齐，存在微小延迟
5. **序列边界**: 使用 `batch_id` 识别连续驾驶场景，不可跨batch构建轨迹
6. **帧率**: 约10Hz，需要45帧 = 4.5秒未来轨迹

## 路径映射关系

| data_info.json 中的路径 | 实际文件位置 |
|------------------------|-------------|
| `vehicle-side/image/000020.jpg` | `cooperative-vehicle-infrastructure-vehicle-side-image/000020.jpg` |
| `infrastructure-side/image/000084.jpg` | `cooperative-vehicle-infrastructure-infrastructure-side-image/000084.jpg` |
| `vehicle-side/calib/novatel_to_world/000020.json` | `cooperative-vehicle-infrastructure/vehicle-side/calib/novatel_to_world/000020.json` |

## 与V2X-VLM代码的对应关系

代码期望从 `ground_truth_trajectories/{frame_id}.npy` 加载GT轨迹,但该目录为空。

**解决方案**: 编写预处理脚本从 `novatel_to_world` 生成轨迹数据。
