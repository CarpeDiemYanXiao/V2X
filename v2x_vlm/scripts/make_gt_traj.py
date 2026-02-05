import json, os, numpy as np
from tqdm import tqdm

# 1. 读官方 meta 文件
info_path = '/root/autodl-tmp/v2x_vlm/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/cooperative/data_info.json'
out_dir   = '/root/autodl-tmp/v2x_vlm/data/ground_truth_trajectories'
os.makedirs(out_dir, exist_ok=True)

info = json.load(open(info_path))

# 2. 以 vehicle 图片序号作为全局帧序号，排序
def vehicle_idx(rec):
    return int(os.path.splitext(os.path.basename(rec['vehicle_image_path']))[0])
info = sorted(info, key=vehicle_idx)

traj_seq = []          # 逐步收集 ego 坐标
last_xy  = None

for rec in tqdm(info, desc='building traj'):
    idx = vehicle_idx(rec)
    lbl_path = os.path.join('/root/autodl-tmp/v2x_vlm/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure',
                            rec['cooperative_label_path'])
    if os.path.exists(lbl_path):
        frame_objs = json.load(open(lbl_path))   # List[dict]
        # 取第一辆 car 作为 ego
        ego = next((o for o in frame_objs if o.get('type') == 'car'), None)
        if ego is not None:
            last_xy = [ego['3d_location']['x'], ego['3d_location']['y']]
    # 如果文件不存在或没 car，沿用 last_xy
    traj_seq.append(last_xy if last_xy else [0., 0.])

# 3. 对每一帧生成未来 45 步（含自己）
for i, rec in enumerate(tqdm(info, desc='save npy')):
    idx = vehicle_idx(rec)
    future = []
    for t in range(i, i+45):
        if t < len(traj_seq):
            future.append(traj_seq[t])
        else:
            future.append(traj_seq[-1])   # 不足时重复最后一帧
    future = np.array(future, dtype=np.float32)   # (45,2)
    np.save(os.path.join(out_dir, f'{idx:06d}.npy'), future)