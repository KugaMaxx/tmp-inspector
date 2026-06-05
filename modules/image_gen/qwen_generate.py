import re
import random
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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
        usecols=["id", "category", "xs", "s", "m", "l", "xl"],
        dtype={"id": "int64", "category": "string", "xs": "int64", "s": "int64", "m": "int64", "l": "int64", "xl": "int64"},
        on_bad_lines="error",
    )

    # Build an object pool
    for row in df.itertuples(index=False):
        class_id = int(row.id)
        category = " ".join(str(row.category).strip().lower().split())
        xs_n = max(0, int(row.xs))
        s_n = max(0, int(row.s))
        m_n = max(0, int(row.m))
        l_n = max(0, int(row.l))
        xl_n = max(0, int(row.xl))

        object_pool.extend([(f"[xs, {category}]", class_id)] * xs_n)
        object_pool.extend([(f"[s, {category}]", class_id)] * s_n)
        object_pool.extend([(f"[m, {category}]", class_id)] * m_n)
        object_pool.extend([(f"[l, {category}]", class_id)] * l_n)
        object_pool.extend([(f"[xl, {category}]", class_id)] * xl_n)

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
        return None

    # extract prompt and response parts
    prompt, response = decoded_text.split(";", 1)
    response = response.split(";", 1)[0]

    # extract info
    labels = re.findall(r"\[(.*?)\]", prompt)
    bboxes = re.findall(r"\[(.*?)\]", response)

    if len(labels) == 0 or len(bboxes) == 0 or len(labels) > len(bboxes):
        return None

    # only consider up to the number of labels
    n = len(labels)

    # parse bbox coordinates
    parsed_results = []
    for label, bbox in zip(labels[:n], bboxes[:n]):
        vals = [v.strip() for v in bbox.split(",")]

        # Expecting exactly 4 values: x, y, w, h
        if len(vals) != 4:
            return None
        try:
            x, y, w, h = [float(v) for v in vals]
        except ValueError:
            return None

        # Strict [x, y, w, h] normalized format
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return None

        # append valid bbox
        parsed_results.append([label, [x, y, w, h]])

    return parsed_results


def write_yolo_label(label_path, class_ids, bboxes):
    n = min(len(class_ids), len(bboxes))
    with open(label_path, "w", encoding="utf-8") as f:
        for class_id, (x, y, w, h) in zip(class_ids[:n], bboxes[:n]):
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def prompt_to_object_texts(prompt):
    parts = re.findall(r"\[\s*(xs|s|m|l|xl)\s*,\s*([^\]]+?)\s*\]", prompt, flags=re.IGNORECASE)
    # return [f"{size.lower()} {category.strip()}" for size, category in parts]
    return [f"{category.strip()}" for size, category in parts]


def yolo_to_xyxy(box, resolution):
    x, y, w, h = box
    x1 = (x - w / 2.0) * resolution
    y1 = (y - h / 2.0) * resolution
    x2 = (x + w / 2.0) * resolution
    y2 = (y + h / 2.0) * resolution
    return x1, y1, x2, y2


def color_for_label(label):
    palette = [
        (255, 59, 48),
        (52, 199, 89),
        (0, 122, 255),
        (255, 149, 0),
        (175, 82, 222),
        (255, 45, 85),
        (90, 200, 250),
        (255, 204, 0),
        (48, 209, 88),
        (64, 156, 255),
    ]
    idx = int.from_bytes(label.encode("utf-8"), "little", signed=False) % len(palette)
    return palette[idx]


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


def draw_bboxes_on_image(image, bboxes, outline_width=6):
    preview_img = image.copy()
    draw = ImageDraw.Draw(preview_img)
    font = ImageFont.load_default()
    width, height = preview_img.size

    for label, box in bboxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, width)
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))

        if x2 > x1 and y2 > y1:
            color = color_for_label(label)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)

            text = str(label)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            text_x = x1
            text_y = max(0.0, y1 - text_height - 4)
            draw.rectangle(
                [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4],
                fill=color,
            )
            draw.text((text_x + 3, text_y + 2), text, fill=(255, 255, 255), font=font)

    return preview_img


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    output_root = Path(args.output_dir)
    labels_dir = output_root / "labels"
    condition_images_dir = output_root / "condition_images"
    generated_images_dir = output_root / "images"
    preview_dir = output_root / "preview"

    labels_dir.mkdir(parents=True, exist_ok=True)
    condition_images_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_model)

    # Load GPT-2 model
    gpt_ml = AutoModelForCausalLM.from_pretrained(args.gpt_model).to(args.device)
    gpt_ml.eval()

    # Load Qwen image pipeline
    qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
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

        # Parse generated text to extract bounding boxes
        parsed_results = extract_generated_bboxes(decoded_text)
        if parsed_results is None: continue

        label_path = labels_dir / f"{idx:06d}.txt"
        write_yolo_label(label_path, prompt_class_ids, [bbox for _, bbox in parsed_results])

        # Build condition image from generated bboxes
        condition_img = draw_condition_image([bbox for _, bbox in parsed_results], args.resolution)
        condition_path = condition_images_dir / f"{idx:06d}.jpg"
        condition_img.save(condition_path, "JPEG", quality=95)

        # Build Qwen prompt and generate final image
        object_texts = prompt_to_object_texts(prompt)
        objects_desc = ", ".join(object_texts) if object_texts else "target objects"
        qwen_prompt = args.qwen_prompt.format(objects=objects_desc)

        with torch.inference_mode():
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

        preview_img = draw_bboxes_on_image(generated_img, parsed_results)
        preview_path = preview_dir / f"{idx:06d}.jpg"
        preview_img.save(preview_path, "JPEG", quality=95)


if __name__ == "__main__":
    main()
