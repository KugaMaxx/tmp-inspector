import io
import re
import base64
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import pandas as pd

from diffusers import QwenImageEditPlusPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests
import dashscope
from dashscope import MultiModalConversation


# API Configuration
API_KEY = "Your-API-Key"
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

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
            "replace ONLY the red bounding boxes with: {objects}. "
            "make them appear as if they were always part of the scene. "
            "keep object count and positions aligned with the red boxes. "
            "consistent lighting with surroundings, photorealistic indoor environment. "
            "do not change image size."
        ),
        help="Template for image generation prompt.",
    )
    parser.add_argument(
        "--qwen_negative_prompt",
        type=str,
        default="text, watermark, logo, red outline, red rectangle, distorted",
        help="Negative prompt for Qwen image generation."
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
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
        usecols=["id", "category", "xs", "s", "m", "l", "xl", "description"],
        dtype={"id": "int64", "category": "string", "xs": "int64", "s": "int64", "m": "int64", "l": "int64", "xl": "int64", "description": "string"},
        on_bad_lines="error",
    )

    # Build an object pool
    object_pool = []
    for row in df.itertuples(index=False):
        class_id = int(row.id)
        class_desc = " ".join(str(row.description).strip().lower().split())
        category = " ".join(str(row.category).strip().lower().split())

        xs_n = max(0, int(row.xs))
        s_n = max(0, int(row.s))
        m_n = max(0, int(row.m))
        l_n = max(0, int(row.l))
        xl_n = max(0, int(row.xl))

        object_pool.extend([(f"[xs, {category}]", class_id, class_desc)] * xs_n)
        object_pool.extend([(f"[s, {category}]",  class_id, class_desc)] * s_n)
        object_pool.extend([(f"[m, {category}]",  class_id, class_desc)] * m_n)
        object_pool.extend([(f"[l, {category}]",  class_id, class_desc)] * l_n)
        object_pool.extend([(f"[xl, {category}]", class_id, class_desc)] * xl_n)

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

        prompt = ", ".join([p for p, _, _ in selected]) + " ;"
        class_ids = [cid for _, cid, _ in selected]
        class_desc = [desc for _, _, desc in selected]
        items.append((prompt, class_ids, class_desc))

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


def draw_condition_image(bboxes, resolution, outline_width=24):
    img = Image.new("RGB", (resolution, resolution), color="black")
    draw = ImageDraw.Draw(img)

    for _, bbox in bboxes:
        x, y, w, h = bbox
        left   = int((x - w / 2.0) * resolution)
        top    = int((y - h / 2.0) * resolution)
        right  = int((x + w / 2.0) * resolution)
        bottom = int((y + h / 2.0) * resolution)

        draw.rectangle([left, top, right, bottom], outline="red", width=outline_width)

    return img


def image_to_base64(image):
    """Convert PIL Image to base64 string for API transmission."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def generate_image_with_api(condition_image, prompt, args):
    """Generate image using Qwen API with base64 encoded condition image."""
    # Convert condition image to base64
    image_base64 = image_to_base64(condition_image)

    # Prepare messages for API
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_base64},
                {"text": prompt}
            ]
        }
    ]

    # Determine seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    # Call API
    response = MultiModalConversation.call(
        api_key=API_KEY,
        model=args.qwen_model.replace("api:", ""),
        messages=messages,
        result_format='message',
        stream=False,
        n=1,
        watermark=False,
        negative_prompt=args.qwen_negative_prompt,
        seed=seed,
        steps=args.num_inference_steps,
    )

    # Extract image URL from response
    if response.status_code == 200:
        output = response.output
        if 'choices' in output and len(output['choices']) > 0:
            choice = output['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
                # Content is a list, find the image URL
                for item in content:
                    if isinstance(item, dict) and 'image' in item:
                        image_url = item['image']
                        # Download the image
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            return Image.open(io.BytesIO(img_response.content))

    raise RuntimeError(f"API call failed: {response}")


def draw_bboxes_on_image(image, bboxes, outline_width=12):
    preview_img = image.copy()
    draw = ImageDraw.Draw(preview_img)
    width, height = preview_img.size

    for entry in bboxes:
        # support either plain box [x,y,w,h] or parsed entry [label, [x,y,w,h]]
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], str):
            label, box = entry
        else:
            label = None
            box = entry

        try:
            x, y, w, h = box
        except Exception:
            continue

        # convert normalized yolo to image coordinates (handles non-square images)
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height

        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))

        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=outline_width)

            # draw label if available (expecting format like "xs, category")
            if label:
                parts = [p.strip() for p in str(label).split(",")]
                if len(parts) >= 2:
                    size = parts[0]
                    category = ",".join(parts[1:])
                    text = f"{size} {category}"
                else:
                    text = parts[0]

                # measure text
                try:
                    text_w, text_h = draw.textsize(text)
                except Exception:
                    # fallback
                    text_w, text_h = (len(text) * 6, 12)

                # position text above the bbox if possible, otherwise below
                pad = 4
                tx1 = int(x1)
                ty1 = int(y1) - text_h - pad
                if ty1 < 0:
                    ty1 = int(y2) + pad
                tx2 = tx1 + text_w + pad
                ty2 = ty1 + text_h + pad // 2

                # clamp to image
                tx1 = max(0, tx1)
                ty1 = max(0, ty1)
                tx2 = min(int(width - 1), tx2)
                ty2 = min(int(height - 1), ty2)

                # draw background rectangle and text
                draw.rectangle([tx1, ty1, tx2, ty2], fill="black")
                draw.text((tx1 + 2, ty1 + 1), text, fill="white")

    return preview_img


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
    if args.qwen_model.startswith("api:"):
        print("Using API for Qwen image generation.")
        qwen_pipeline = None
    else:
        print("Loading Qwen image edit model...")
        qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.qwen_model,
            torch_dtype=torch.bfloat16,
        )

        if args.cpu_offload and args.device.startswith("cuda"):
            qwen_pipeline.enable_model_cpu_offload()
        else:
            qwen_pipeline = qwen_pipeline.to(args.device)

    for idx, (gpt_prompt, class_ids, class_desc) in enumerate(gpt_items):
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

        # Prepare Qwen prompt and condition image
        qwen_prompt = args.qwen_prompt_template.format(
            objects=bboxes[0][0].split(",")[1].strip(),
            description="; ".join(class_desc)
        )
        condition_image = draw_condition_image(bboxes, args.resolution)

        # Generate image with Qwen
        if qwen_pipeline is not None:
            with torch.inference_mode():
                generated_image = qwen_pipeline(
                    image=[condition_image],
                    prompt=qwen_prompt,
                    negative_prompt=args.qwen_negative_prompt,
                    generator=torch.manual_seed(args.seed if args.seed is not None else random.randint(0, int(1e6))),
                    num_inference_steps=args.num_inference_steps,
                    cfg_scale=args.cfg_scale,
                ).images[0]
        else:
            try:
                generated_image = generate_image_with_api(condition_image, qwen_prompt, args)
            except Exception as e:
                print(f"API call failed: {e}. Skipping this item.")
                continue

        # Save YOLO file
        ## label file
        with open(label_dir / f"{idx:10d}.txt", "w") as f:
            for class_id, (_, bbox) in zip(class_ids, bboxes):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        ## image file
        generated_image.save(image_dir / f"{idx:10d}.png")

        ## preview file
        preview_image = draw_bboxes_on_image(generated_image, bboxes)
        preview_image.save(preview_dir / f"{idx:10d}.png")


if __name__ == "__main__":
    main()
