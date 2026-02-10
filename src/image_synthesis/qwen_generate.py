import yaml
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import pandas as pd

from diffusers import QwenImageEditPlusPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class_config",
        type=str,
        default="config/qwen_class_tmp.csv",
        help="Path to the global configuration file.",
    )

    # Qwen Image
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Path to the Qwen image editing model or model identifier from huggingface.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution for image generation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "replace ONLY the red bounding box to the {category}. "
            "make it appear as if it was always a part of the scene. "
            "consistent illumination with the surroundings. "
            "do not change the size of the image. "
            "replace the black background with a photorealistic indoor scene. "
            "ceiling-mounted smoke detector, "
            "circular white device with radial ventilation openings, "
            "compact plastic body, small status LED, fixed on indoor ceiling surface."
        ),
        help="Template for image generation prompt. Use {category} as placeholder.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "text, watermark, logo, red outline, red rectangle, object outside rectangle, distorted"
        ),
        help="Negative prompt for image generation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for image generation.",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale for image generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for image generation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help='Device to run the pipeline on (e.g., "cuda" or "cpu").',
    )

    # Generate settings
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin to ensure the rectangle does not go out of bounds.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen-image-fire-smoke_detector",
        help="Directory to save generated images.",
    )

    # CPU offload
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        default=True,
        help="Enable Diffusers CPU offloading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Create YOLO dataset directory structure
    output_path = Path(args.output_dir)
    
    train_images_path = output_path / "images" / "train"
    train_images_path.mkdir(parents=True, exist_ok=True)

    train_labels_path = output_path / "labels" / "train"
    train_labels_path.mkdir(parents=True, exist_ok=True)
    
    preview_path = output_path / "preview" / "train"
    preview_path.mkdir(parents=True, exist_ok=True)

    # Class configuration
    class_config = pd.read_csv(args.class_config)
    
    # Create data.yaml for YOLO
    data_yaml = {
        'train': str(train_images_path.absolute()),
        'val': str(train_images_path.absolute()),
        'nc': len(class_config['id']),
        'names': class_config['category'].tolist()
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # Load Qwen Image Edit Plus Pipeline
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
    )
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)
    
    for index, entry in class_config.iterrows():
        prompt = args.prompt.format(category=entry['category'].replace('_', ' '))
        negative_prompt = args.negative_prompt

        for n in range(entry['count']):
            # Draw image with bounding box
            bbox_img = Image.new("RGB", (args.resolution, args.resolution), color=(0, 0, 0))
            bbox_draw = ImageDraw.Draw(bbox_img)
            
            # Random rectangle size
            scale = random.uniform(entry['scale_min'], entry['scale_max'])
            h = scale * args.resolution
            w = scale * args.resolution * entry['ratio']
            
            # Random rectangle position
            max_x = args.resolution - w - args.margin
            max_y = args.resolution - h - args.margin
            assert max_x > args.margin and max_y > args.margin, \
            "Rectangle size too large for the given resolution and margin."
            
            # Draw rectangle
            x = random.uniform(args.margin, max_x)
            y = random.uniform(args.margin, max_y)
            bbox_draw.rectangle([x, y, x + w, y + h], outline="red", width=6)

            # Generate image
            result = pipeline(
                image=[bbox_img],
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
            ).images[0]
                
            # Create unique filename
            image_filename = f"{entry['category']}_{n:06d}.jpg"
            image_path = train_images_path / image_filename
            result.save(image_path, 'JPEG', quality=95)
            
            # Create and save preview image (result with bbox drawn on it)
            preview_img = result.copy()
            preview_draw = ImageDraw.Draw(preview_img)
            preview_draw.rectangle([x, y, x + w, y + h], outline="red", width=6)
            preview_filename = f"{entry['category']}_{n:06d}_preview.jpg"
            preview_img.save(preview_path / preview_filename, 'JPEG', quality=95)
            
            # Convert bbox to YOLO format (normalized coordinates)
            # YOLO format: class_id center_x center_y width height
            center_x = (x + w / 2) / args.resolution
            center_y = (y + h / 2) / args.resolution
            norm_width = w / args.resolution
            norm_height = h / args.resolution
            
            # Save YOLO annotation file
            label_filename = f"{entry['category']}_{n:06d}.txt"
            label_path = train_labels_path / label_filename
            
            with open(label_path, 'w') as f:
                f.write(f"{entry['id']} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
