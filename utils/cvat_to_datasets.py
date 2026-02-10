import os
import math
import argparse

import pandas as pd
from datasets import Dataset

import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_size(area, small=[0, 0.004], medium=[0.004, 0.036]):
    if area < small[1]:
        return 'small'
    elif area < medium[1]:
        return 'medium'
    else:
        return 'large'


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="path to annotations XML file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="output path for parquet dataset"
    )
    args = parser.parse_args()

    print(f"Loading XML annotations from: {args.input_path}")

    # Parse annotations from XML
    root = ET.parse(args.input_path).getroot()
    labels = root.find('meta').find('task').find('labels').findall('label')

    print(f"Found {len(root.findall('image'))} images, {len(labels)} categories")
    
    # Process annotations
    print("Converting annotations to parquet...")

    texts = []
    for image in tqdm(root.findall('image'), desc="Processing"):
        data = {
            "id": int(image.get('id')),
            "name": image.get('name'),
            "width": int(image.get('width')),
            "height": int(image.get('height')),
        }

        text = {
            'label': [],
            'size': [],
            'coordinates': []
        }
        
        for entry in image.findall('box'):
            # Extract box information
            label = entry.get('label')
            xtl = float(entry.get('xtl')) / data['width']
            ytl = float(entry.get('ytl')) / data['height']
            xbr = float(entry.get('xbr')) / data['width']
            ybr = float(entry.get('ybr')) / data['height']

            # Calculate center, size, and aspect ratio
            cx = (xtl + xbr) / 2
            cy = (ytl + ybr) / 2
            size = get_size((xbr - xtl) * (ybr - ytl))
            ratio = math.log((xbr - xtl) / (ybr - ytl + 1e-6))

            # Update text data
            text['label'].append(label)
            text['size'].append(size)
            text['coordinates'].append([cx, cy, ratio])

        # Format: size label, size label, ... ; [cx, cy, ratio], [cx, cy, ratio], ...
        if len(image.findall('box')) > 0:
            prompt = [f'[{size}, {label}]' for size, label in zip(text['size'], text['label'])]
            response = [f'[{cx:.3f}, {cy:.3f}, {ratio:.3f}]' for cx, cy, ratio in text['coordinates']]
            texts.append(', '.join(prompt) + ' ; ' + ', '.join(response))

    # Create output directory
    print(f"Saving parquet dataset to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save to output directory
    df = pd.DataFrame({'text': texts})
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.to_parquet(args.output_path)

    print("Conversion completed successfully!")
