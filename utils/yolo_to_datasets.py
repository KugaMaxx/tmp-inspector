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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO annotations into a huggingface dataset."
        )
    )
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
    args = parser.parse_args()

    # Load dataset info from YOLODataset
    print(f"Loading dataset from: {args.input_path}")
    dataset_info = check_det_dataset(args.input_path)
    
    class_names = dataset_info.get("names", {})

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
        
        print(f"Processing {split} split, {len(dataset.labels)} images...")
        
        hf_data = {
            "image": [],
            "objects": [],
            "width": [],
            "height": []
        }
        
        # Iterate through the dataset and convert to HuggingFace format
        for i in tqdm(range(len(dataset))):
            # Get the image and annotations
            item = dataset[i]
            
            # Convert image tensor back to PIL format
            img_pil = TF.to_pil_image(item["img"])
            
            # Retrieve classes and boxes
            cls_ids = item["cls"].flatten().tolist()
            bboxes = item["bboxes"].tolist() if len(item["bboxes"]) > 0 else []
            
            # Map class IDs to category names
            categories = [int(c) for c in cls_ids]
            category_names = [class_names.get(int(c), str(c)) for c in cls_ids]
            
            # Append data to HuggingFace dataset format
            hf_data["image"].append(img_pil)
            hf_data["width"].append(img_pil.width) 
            hf_data["height"].append(img_pil.height)
            hf_data["objects"].append({
                "bbox": bboxes, 
                "category": categories,
                "category_name": category_names
            })
            
        # Convert to HuggingFace dataset and save as parquet
        hf_dataset = datasets.Dataset.from_dict(hf_data)
        hf_dataset = hf_dataset.cast_column("image", datasets.Image())
        
        # Save the dataset in parquet format
        output_file = Path(args.output_dir) / f"{split}.parquet"
        hf_dataset.to_parquet(output_file)
        print(f"Saved {split} to {output_file}")


if __name__ == "__main__":
    main()
