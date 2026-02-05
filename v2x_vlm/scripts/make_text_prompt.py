import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

# =========================
# 路径配置
# =========================
INFO_PATH = '/root/autodl-tmp/v2x_vlm/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/cooperative/data_info.json'
VEHICLE_DIR = '/root/autodl-tmp/v2x_vlm/data/vehicle_images'
INFRA_DIR = '/root/autodl-tmp/v2x_vlm/data/infrastructure_images'
OUT_DIR = '/root/autodl-tmp/v2x_vlm/data/text_prompts'

# ⚠️ 本地 Florence-2 模型路径
FLORENCE2_DIR = '/root/autodl-tmp/models/florence2/Florence-2-large'

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 加载 Florence-2（本地离线）
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = AutoProcessor.from_pretrained(
    FLORENCE2_DIR,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    FLORENCE2_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).eval().to(device)

# =========================
# 工具函数
# =========================
def detailed_caption(img_path: str) -> str:
    image = Image.open(img_path).convert('RGB')

    inputs = processor(
        text="<DETAILED_CAPTION>",
        images=image,
        return_tensors="pt"
    )

    # 一次完成 dtype + device 转换
    inputs = {k: v.half().to(device) if k == 'pixel_values' else v.to(device)
              for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3
        )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def build_prompt(desc, curr_xy):
    return (
        f"<SCENE_DESCRIPTION>: {desc}\n"
        f"<CURRENT_POSITION>: The current position of the ego vehicle is "
        f"({curr_xy[0]:.2f}, {curr_xy[1]:.2f}).\n"
        f"<OBJECTIVE>: Please predict the ego vehicle positions over next 45 timestamps."
    )

# =========================
# 按 vehicle idx 排序
# =========================
def vehicle_idx(rec):
    return int(os.path.splitext(os.path.basename(rec['vehicle_image_path']))[0])

with open(INFO_PATH, 'r') as f:
    info = json.load(f)

info = sorted(info, key=vehicle_idx)

# =========================
# 主循环
# =========================
for rec in tqdm(info, desc='Florence-2 captioning'):
    idx = vehicle_idx(rec)

    vimg = os.path.join(VEHICLE_DIR, f'{idx:06d}.jpg')
    iimg = os.path.join(INFRA_DIR, f'{idx:06d}.jpg')

    if not (os.path.exists(vimg) and os.path.exists(iimg)):
        continue

    # Florence-2 双视角描述
    vcap = detailed_caption(vimg)
    icap = detailed_caption(iimg)

    desc = (
        "The image on the left is captured by the ego vehicle's front camera, "
        "and the image on the right is captured by the roadside camera. "
        f"{vcap} {icap}"
    )

    # 当前 ego 坐标
    lbl_path = os.path.join(
        '/root/autodl-tmp/v2x_vlm/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure',
        rec['cooperative_label_path']
    )

    if os.path.exists(lbl_path):
        frame_objs = json.load(open(lbl_path))
        ego = next(
            (o for o in frame_objs if isinstance(o, dict) and o.get('type') == 'car'),
            None
        )
        curr_xy = [ego['3d_location']['x'], ego['3d_location']['y']] if ego else [0.0, 0.0]
    else:
        curr_xy = [0.0, 0.0]

    # 写入 prompt
    with open(os.path.join(OUT_DIR, f'{idx:06d}.txt'), 'w') as f:
        f.write(build_prompt(desc, curr_xy))
