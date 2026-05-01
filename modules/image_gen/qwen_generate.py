import re
import random
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image, ImageDraw

from diffusers import QwenImageEditPlusPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained GPT-2 on prompts and save YOLO labels.")
    # GPT-2 generation settings
    parser.add_argument(
        "--gpt_model", 
        type=str, 
        required=True,
        help="Path to trained GPT-2 model."
    )
    parser.add_argument(
        "--prompt_csv", 
        type=str, 
        required=True,
        help="CSV file with columns: id,category,small,medium,large."
    )
    parser.add_argument(
        "--max_objects_per_prompt",
        type=int,
        default=1,
        help="Maximum number of objects in one generated prompt. <=0 means no limit."
    )

    # Qwen image generation settings
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Path to Qwen image editing model or model id on huggingface."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square image resolution for condition and generated images."
    )
    parser.add_argument(
        "--qwen_prompt",
        type=str,
        default=(
            "replace ONLY the red bounding boxes with: {objects}. "
            "make them appear as if they were always part of the scene. "
            "keep object count and positions aligned with the red boxes. "
            "consistent lighting with surroundings, photorealistic indoor environment. "
            "do not change image size."
        ),
        help="Prompt template for Qwen image generation. Use {objects} as placeholder."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="text, watermark, logo, red outline, red rectangle, distorted",
        help="Negative prompt for Qwen image generation."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for Qwen image generation."
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale for Qwen image generation."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for Qwen image generation."
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        default=True,
        help="Enable Diffusers CPU offloading for Qwen pipeline."
    )

    # Inference settings
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        help='Inference device, e.g. "cuda" or "cpu".'
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256, 
        help="Maximum new tokens."
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true", 
        help="Enable sampling during generation."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.9, 
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p sampling."
    )

    # Output settings
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='Qwen-Fire',
        help="Directory to save generated YOLO label txt files."
    )

    return parser.parse_args()


def build_prompts(prompt_csv, max_objects_per_prompt):
    object_pool = []
    df = pd.read_csv(
        prompt_csv,
        usecols=["id", "category", "small", "medium", "large"],
        dtype={"id": "int64", "category": "string", "small": "int64", "medium": "int64", "large": "int64"},
        on_bad_lines="error",
    )

    # Build an object pool
    for row in df.itertuples(index=False):
        class_id = int(row.id)
        category = " ".join(str(row.category).strip().lower().split())
        small_n = max(0, int(row.small))
        medium_n = max(0, int(row.medium))
        large_n = max(0, int(row.large))

        object_pool.extend([(f"[small, {category}]", class_id)] * small_n)
        object_pool.extend([(f"[medium, {category}]", class_id)] * medium_n)
        object_pool.extend([(f"[large, {category}]", class_id)] * large_n)

    # Randomly consume object pool to create multiple random combinations
    items = []
    while len(object_pool) > 0:
        max_n = len(object_pool) if max_objects_per_prompt <= 0 else min(max_objects_per_prompt, len(object_pool))
        n = random.randint(1, max_n)

        selected_indices = random.sample(range(len(object_pool)), n)
        selected = []
        for i in sorted(selected_indices, reverse=True):
            selected.append(object_pool.pop(i))
        random.shuffle(selected)

        prompt = ", ".join([p for p, _ in selected]) + " ;"
        class_ids = [cid for _, cid in selected]
        items.append((prompt, class_ids))

    return items


def extract_generated_bboxes(decoded_text):
    if ";" not in decoded_text:
        return []
    
    BBOX_PATTERN = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )

    response_part = decoded_text.split(";", 1)[1]
    boxes = []
    for x_str, y_str, w_str, h_str in BBOX_PATTERN.findall(response_part):
        x, y, w, h = float(x_str), float(y_str), float(w_str), float(h_str)
        # keep only valid YOLO normalized values
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
            boxes.append((x, y, w, h))
    return boxes


def write_yolo_label(label_path, class_ids, bboxes):
    n = min(len(class_ids), len(bboxes))
    with open(label_path, "w", encoding="utf-8") as f:
        for class_id, (x, y, w, h) in zip(class_ids[:n], bboxes[:n]):
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def prompt_to_object_texts(prompt):
    parts = re.findall(r"\[\s*(small|medium|large)\s*,\s*([^\]]+?)\s*\]", prompt, flags=re.IGNORECASE)
    return [f"{size.lower()} {category.strip()}" for size, category in parts]


def yolo_to_xyxy(box, resolution):
    x, y, w, h = box
    x1 = (x - w / 2.0) * resolution
    y1 = (y - h / 2.0) * resolution
    x2 = (x + w / 2.0) * resolution
    y2 = (y + h / 2.0) * resolution
    return x1, y1, x2, y2


def draw_condition_image(bboxes, resolution, outline_width=6):
    img = Image.new("RGB", (resolution, resolution), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    for box in bboxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, resolution)
        x1 = max(0.0, min(float(resolution - 1), x1))
        y1 = max(0.0, min(float(resolution - 1), y1))
        x2 = max(0.0, min(float(resolution - 1), x2))
        y2 = max(0.0, min(float(resolution - 1), y2))

        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=outline_width)

    return img


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    output_root = Path(args.output_dir)
    labels_dir = output_root / "labels"
    condition_images_dir = output_root / "condition_images"
    generated_images_dir = output_root / "images"

    labels_dir.mkdir(parents=True, exist_ok=True)
    condition_images_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_model)

    # Load GPT-2 model
    gpt_ml = AutoModelForCausalLM.from_pretrained(args.gpt_model).to(args.device)
    gpt_ml.eval()

    # Load Qwen image pipeline
    qwen_dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.qwen_model,
        torch_dtype=qwen_dtype,
    )

    if args.cpu_offload and args.device.startswith("cuda"):
        qwen_pipeline.enable_model_cpu_offload()
    else:
        qwen_pipeline = qwen_pipeline.to(args.device)

    # Build prompts randomly
    prompt_items = build_prompts(args.prompt_csv, args.max_objects_per_prompt)

    # Process prompts and generate labels
    for idx, (prompt, prompt_class_ids) in enumerate(prompt_items):

        # Generate bounding boxes with GPT-2
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = gpt_ml.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 
        bboxes = extract_generated_bboxes(decoded_text)
        n = min(len(prompt_class_ids), len(bboxes))
        aligned_class_ids = prompt_class_ids[:n]
        aligned_bboxes = bboxes[:n]

        label_path = labels_dir / f"{idx:06d}.txt"
        write_yolo_label(label_path, aligned_class_ids, aligned_bboxes)

        # Build condition image from generated bboxes
        condition_img = draw_condition_image(aligned_bboxes, args.resolution)
        condition_path = condition_images_dir / f"{idx:06d}.jpg"
        condition_img.save(condition_path, "JPEG", quality=95)

        # Build Qwen prompt and generate final image
        object_texts = prompt_to_object_texts(prompt)
        object_texts = object_texts[:n]
        objects_desc = ", ".join(object_texts) if object_texts else "target objects"
        qwen_prompt = args.qwen_prompt.format(objects=objects_desc)

        generated_img = qwen_pipeline(
            image=[condition_img],
            prompt=qwen_prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=1,
        ).images[0]

        generated_path = generated_images_dir / f"{idx:06d}.jpg"
        generated_img.save(generated_path, "JPEG", quality=95)


if __name__ == "__main__":
    main()
