"""
场景描述生成脚本

使用 Florence-2-large 的 <DETAILED_CAPTION> 功能生成场景描述
用于构建文本提示 (Text Prompt E)

论文: "The descriptions are obtained using the built-in <DETAILED_CAPTION> API option"
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def generate_scene_descriptions(
    vehicle_image_dir,
    infra_image_dir,
    output_path,
    device="cuda",
    batch_size=1,
    max_samples=None,
    cache_dir=None
):
    """
    使用Florence-2生成场景描述
    
    论文描述:
    - Scene description resulted from the ability of VLM to understand and interpret 
      the complex driving environment
    - Using the built-in <DETAILED_CAPTION> API option
    
    Args:
        vehicle_image_dir: 车端图像目录
        infra_image_dir: 路侧图像目录  
        output_path: 输出JSON路径
        device: 计算设备
        batch_size: 批次大小
        max_samples: 最大样本数 (用于测试)
        cache_dir: 模型缓存目录
    """
    print("加载 Florence-2-large 模型...")
    
    # 设置模型缓存目录 (默认为项目目录下的 pretrained_models)
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "pretrained_models"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model cache directory: {cache_dir}")
    
    # 加载模型 (使用 attn_implementation="eager" 避免 SDPA 兼容性问题)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=str(cache_dir),
        attn_implementation="eager",
        local_files_only=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        local_files_only=True
    )
    
    model.eval()
    
    vehicle_image_dir = Path(vehicle_image_dir)
    infra_image_dir = Path(infra_image_dir)
    
    # 获取图像列表
    vehicle_images = sorted(vehicle_image_dir.glob("*.jpg"))
    infra_images = sorted(infra_image_dir.glob("*.jpg"))
    
    if max_samples:
        vehicle_images = vehicle_images[:max_samples]
    
    print(f"车端图像: {len(vehicle_images)}")
    print(f"路侧图像: {len(infra_images)}")
    
    # 构建路侧图像映射
    infra_map = {img.stem: img for img in infra_images}
    
    descriptions = {
        'vehicle': {},
        'infrastructure': {}
    }
    
    # 生成车端描述
    print("\n生成车端场景描述...")
    for img_path in tqdm(vehicle_images, desc="Vehicle"):
        frame_id = img_path.stem
        
        try:
            image = Image.open(img_path).convert("RGB")
            description = generate_single_description(
                model, processor, image, device
            )
            descriptions['vehicle'][frame_id] = description
        except Exception as e:
            print(f"  跳过 {frame_id}: {e}")
            descriptions['vehicle'][frame_id] = "A driving scene."
    
    # 生成路侧描述 (去重)
    print("\n生成路侧场景描述...")
    processed_infra = set()
    
    for img_path in tqdm(infra_images[:len(vehicle_images)], desc="Infrastructure"):
        frame_id = img_path.stem
        
        if frame_id in processed_infra:
            continue
        
        try:
            image = Image.open(img_path).convert("RGB")
            description = generate_single_description(
                model, processor, image, device
            )
            descriptions['infrastructure'][frame_id] = description
            processed_infra.add(frame_id)
        except Exception as e:
            print(f"  跳过 {frame_id}: {e}")
            descriptions['infrastructure'][frame_id] = "An intersection scene."
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存到: {output_path}")
    print(f"  - 车端描述: {len(descriptions['vehicle'])}")
    print(f"  - 路侧描述: {len(descriptions['infrastructure'])}")
    
    return descriptions


def generate_single_description(model, processor, image, device):
    """
    为单张图像生成描述
    
    使用 <DETAILED_CAPTION> prompt
    """
    prompt = "<DETAILED_CAPTION>"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)
    
    # 确保 pixel_values 类型与模型一致 (float16 on CUDA)
    if device == "cuda" and inputs.get("pixel_values") is not None:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    
    with torch.no_grad():
        # 使用贪婪解码避免 beam search 与 Florence-2 KV cache 不兼容问题
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,  # 禁用 beam search
            use_cache=False,  # 禁用 KV cache 避免兼容性问题
            early_stopping=False  # 显式禁用以避免警告
        )
    
    description = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 清理输出 (移除prompt前缀)
    if description.startswith("<DETAILED_CAPTION>"):
        description = description[len("<DETAILED_CAPTION>"):].strip()
    
    return description


def construct_text_prompt(
    vehicle_description,
    infra_description,
    ego_position,
    task_type="trajectory_planning"
):
    """
    构建完整的文本提示
    
    论文描述 (Section 4.1):
    The text prompt E includes:
    1. Scene description - from VLM
    2. Current position of ego vehicle - from GPS/IMU  
    3. Explicit planning task description
    
    Args:
        vehicle_description: 车端场景描述
        infra_description: 路侧场景描述
        ego_position: (x, y) 自车世界坐标位置
        task_type: 任务类型
    
    Returns:
        完整的文本提示
    """
    prompt = f"""Vehicle View: {vehicle_description}

Infrastructure View: {infra_description}

Current Ego Vehicle Position: x={ego_position[0]:.2f}m, y={ego_position[1]:.2f}m

Task: Plan the future trajectory for the ego vehicle over the next 4.5 seconds. 
Output a sequence of 45 waypoints (x, y) at 10Hz in the ego-centric coordinate system, 
where the current position is the origin and the vehicle heading is the positive x-axis."""

    return prompt


def main():
    parser = argparse.ArgumentParser(description='生成场景描述')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='数据集根目录')
    parser.add_argument('--output', type=str, 
                        default='../data/scene_descriptions.json',
                        help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数 (用于测试)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='模型缓存目录 (默认: pretrained_models/)')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    vehicle_image_dir = data_root / "cooperative-vehicle-infrastructure-vehicle-side-image"
    infra_image_dir = data_root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
    
    if not vehicle_image_dir.exists():
        print(f"❌ 车端图像目录不存在: {vehicle_image_dir}")
        return
    
    if not infra_image_dir.exists():
        print(f"❌ 路侧图像目录不存在: {infra_image_dir}")
        return
    
    print("=" * 60)
    print("V2X-VLM 场景描述生成器")
    print("=" * 60)
    
    generate_scene_descriptions(
        vehicle_image_dir=vehicle_image_dir,
        infra_image_dir=infra_image_dir,
        output_path=args.output,
        device=args.device,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir
    )
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
