import json, os, collections, random
info = json.load(open('/root/autodl-tmp/v2x_vlm/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/cooperative/data_info.json'))
os.makedirs('/root/autodl-tmp/v2x_vlm/data/split', exist_ok=True)

# 按 8:1:1 简单随机切分（官方无 split 字段时）
random.seed(0)
all_ids = [os.path.splitext(os.path.basename(r['vehicle_image_path']))[0] for r in info]
random.shuffle(all_ids)
n = len(all_ids)
train_ids = all_ids[:int(0.8*n)]
val_ids   = all_ids[int(0.8*n):int(0.9*n)]
test_ids  = all_ids[int(0.9*n):]

for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    with open(f'data/split/{name}.txt', 'w') as f:
        f.write('\n'.join(ids))