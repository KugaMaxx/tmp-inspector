import re
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
import pandas as pd

from diffusers import QwenImageEditPlusPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="AI Image Generator.")
    # GPT-2 generation settings
    parser.add_argument(
        "--gpt_model", 
        type=str, 
        required=True,
        help="Path to trained GPT-2 model."
    )
    parser.add_argument(
        "--gpt_prompt_config", 
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
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256, 
        help="Maximum new tokens."
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true", 
        default=True,
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

    # Qwen immage generation settings
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
        "--qwen_prompt_template",
        type=str,
        default=(
            "Replace the red bounding box with {objects}. "
            "The placeholder {objects} must only appear within the red box and match the box in dimensions. "
            "A real-scene background shall be adopted."
        ),
        help="Template for image generation prompt.",
    )
    parser.add_argument(
        "--qwen_negative_prompt",
        type=str,
        default=(
            " "
        ),
        help="Negative prompt for image generation.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for Qwen image generation."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps for Qwen image generation."
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        default=True,
        help="Enable Diffusers CPU offloading for Qwen pipeline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for API generation. If None, uses random seed per image.",
    )

    # Common settings
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        help='Inference device, e.g. "cuda" or "cpu".'
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='Qwen-Fire',
        help="Directory to save generated YOLO label txt files."
    )

    return parser.parse_args()


def prepare_gpt_prompts(args):
    df = pd.read_csv(
        args.gpt_prompt_config,
        usecols=["id", "category", "xs", "s", "m", "l", "xl"],
        dtype={"id": "int64", "category": "string", "xs": "int64", "s": "int64", "m": "int64", "l": "int64", "xl": "int64"},
        on_bad_lines="error",
    )

    # Build an object pool
    object_pool = []
    for row in df.itertuples(index=False):
        class_id = int(row.id)
        category = " ".join(str(row.category).strip().lower().split())

        xs_n = max(0, int(row.xs))
        s_n = max(0, int(row.s))
        m_n = max(0, int(row.m))
        l_n = max(0, int(row.l))
        xl_n = max(0, int(row.xl))

        object_pool.extend([(f"[xs, {category}]", class_id)] * xs_n)
        object_pool.extend([(f"[s, {category}]",  class_id)] * s_n)
        object_pool.extend([(f"[m, {category}]",  class_id)] * m_n)
        object_pool.extend([(f"[l, {category}]",  class_id)] * l_n)
        object_pool.extend([(f"[xl, {category}]", class_id)] * xl_n)

    # Randomly consume object pool to create multiple random combinations
    items = []
    while len(object_pool) > 0:
        max_n = (
            len(object_pool)
            if args.max_objects_per_prompt <= 0
            else min(args.max_objects_per_prompt, len(object_pool))
        )
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


def extract_bboxes_from_text(output_text):
    if ";" not in output_text:
        return None

    # extract prompt and response parts
    prompt, response = output_text.split(";", 1)
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


# Palette of visually distinct colors. Each entry maps a color name to its RGB
# value so the same name can be referenced in the Qwen prompt.
COLOR_PALETTE = [
    ("red",     (255, 0, 0)),
    ("green",   (0, 200, 0)),
    ("blue",    (0, 0, 255)),
    ("yellow",  (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan",    (0, 255, 255)),
    ("orange",  (255, 128, 0)),
    ("purple",  (140, 0, 255)),
    ("pink",    (255, 105, 180)),
    ("lime",    (170, 255, 0)),
    ("teal",    (0, 160, 160)),
    ("brown",   (150, 75, 0)),
]


def allocate_bbox_colors(n):
    palette = COLOR_PALETTE.copy()
    random.shuffle(palette)
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def draw_condition_image(bboxes, resolution, colors, alpha=64, draw_labels=False, background=None):
    if background is None:
        base = Image.new("RGBA", (resolution, resolution), color=(0, 0, 0, 255))
    else:
        base = background.convert("RGBA")

    width, height = base.size
    overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=24)

    for (label, bbox), (_, rgb) in zip(bboxes, colors):
        x, y, w, h = bbox
        left   = int((x - w / 2.0) * width)
        top    = int((y - h / 2.0) * height)
        right  = int((x + w / 2.0) * width)
        bottom = int((y + h / 2.0) * height)

        # semi-transparent mask fill
        draw.rectangle([left, top, right, bottom], fill=(*rgb, alpha))

        if draw_labels and label is not None:
            (text_width, text_height), (_, _) = font.font.getsize(label)
            text_x = left if left + text_width < width else width - text_width
            text_y = top if top + text_height < height else height - text_height
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


def main():
    # Parse arguments
    args = parse_args()

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Load GPT-2 model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.gpt_model)
    gpt_model = AutoModelForCausalLM.from_pretrained(args.gpt_model).to(args.device)
    gpt_model.eval()

    # Prepare GPT prompts
    print("Preparing GPT prompts...")
    gpt_items = prepare_gpt_prompts(args)

    # Load Qwen image edit model
    print("Loading Qwen image edit model...")
    qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
    )

    if args.cpu_offload and args.device.startswith("cuda"):
        qwen_pipeline.enable_model_cpu_offload()
    else:
        qwen_pipeline = qwen_pipeline.to(args.device)

    for idx, (gpt_prompt, class_ids) in enumerate(gpt_items):
        print(idx)

        # Generate bounding boxes with GPT-2
        input_ids = gpt_tokenizer(gpt_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = gpt_model.generate(
                **input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=gpt_tokenizer.eos_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
            )[0]
        output_text = gpt_tokenizer.decode(output_ids, skip_special_tokens=True)

        # Extract bounding boxes from GPT output
        bboxes = extract_bboxes_from_text(output_text)
        if bboxes is None:
            print("Failed to extract valid bounding boxes from GPT output. Skipping this item.")
            continue

        # Assign a unique color to each bbox
        colors = allocate_bbox_colors(len(bboxes))

        # Prepare Qwen prompt and condition image
        qwen_prompt = " ".join(
            args.qwen_prompt.format(
                objects=label.split(",")[1].strip(),
                color=color,
            )
            for (label, bbox), (color, rgb) in zip(bboxes, colors)
        )
        qwen_negative_prompt = args.qwen_negative_prompt
        condition_image = draw_condition_image(
            bboxes, args.resolution, colors
        )

        # Generate image with Qwen
        with torch.inference_mode():
            generated_image = qwen_pipeline(
                image=[condition_image],
                prompt=qwen_prompt,
                negative_prompt=qwen_negative_prompt,
                generator=torch.manual_seed(args.seed if args.seed is not None else random.randint(0, int(1e6))),
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.cfg_scale,
            ).images[0]

        # Save YOLO file
        file_name = f"{idx:010d}"

        ## label file
        with open(label_dir / f"{file_name}.txt", "w") as f:
            for class_id, (_, bbox) in zip(class_ids, bboxes):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        ## image file
        generated_image.save(image_dir / f"{file_name}.png")

        ## preview file: labeled masks overlaid on the AI result
        preview_image = draw_condition_image(
            bboxes, args.resolution, colors,
            draw_labels=True, background=generated_image,
        )
        preview_image.save(preview_dir / f"{file_name}.png")


if __name__ == "__main__":
    main()
