import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import datasets
import torchvision.transforms.functional as TF

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO annotations into a huggingface dataset.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory containing YOLO label txt files and corresponding images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted dataset (parquet format)."
    )
    parser.add_argument(
        "--img_sz",
        type=int,
        default=512,
        help="Image size used by YOLODataset preprocessing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=15000,
        help="Number of samples kept in memory before flushing to a parquet shard.",
    )
    return parser.parse_args()


def generate_examples(dataset, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        item = dataset[i]

        # Convert image tensor back to PIL format
        img_pil = TF.to_pil_image(item["img"])

        # Retrieve classes and boxes
        cls_ids = item["cls"].flatten().tolist()
        bboxes = item["bboxes"].tolist() if len(item["bboxes"]) > 0 else []

        # Map class IDs to category names
        class_names = dataset.data['names']
        categories = [int(c) for c in cls_ids]
        category_names = [class_names.get(int(c), str(c)) for c in cls_ids]

        yield {
            "image": img_pil,
            "width": img_pil.width,
            "height": img_pil.height,
            "objects": {
                "bbox": bboxes,
                "category": categories,
                "category_name": category_names,
            },
        }


def main():
    # Parse arguments
    args = parse_args()

    # Load dataset info from YOLODataset
    print(f"Loading dataset from: {args.input_path}")
    dataset_info = check_det_dataset(args.input_path)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in dataset_info: continue

        dataset = YOLODataset(
            img_path=dataset_info[split],
            imgsz=args.img_sz,
            data=dataset_info,
            task='detect',
            augment=False,
            rect=False
        )
        
        print(f"\nProcessing {split} split, {len(dataset.labels)} images...")
        shard_idx = 0
        for start_idx in range(0, len(dataset), args.chunk_size):
            end_idx = min(start_idx + args.chunk_size, len(dataset))
            hf_dataset = datasets.Dataset.from_generator(
                generate_examples,
                gen_kwargs={
                    "dataset": dataset,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                },
            )
            hf_dataset = hf_dataset.cast_column("image", datasets.Image())

            output_file = output_dir / f"{split}-{shard_idx:05d}.parquet"
            hf_dataset.to_parquet(output_file)
            shard_idx += 1

        print(f"Finished {split}, total shards: {shard_idx}")


if __name__ == "__main__":
    main()
