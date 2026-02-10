import os
import argparse
from datetime import datetime

from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_size(area, small=[0, 0.004], medium=[0.004, 0.036]):
    """Determine object size based on area"""
    if area < small[1]:
        return 'small'
    elif area < medium[1]:
        return 'medium'
    else:
        return 'large'


def create_cvat_xml(coco):
    """Convert COCO data to CVAT XML format"""
    
    # Create root element
    annotations = ET.Element("annotations")
    
    # Add version information
    version = ET.SubElement(annotations, "version")
    version.text = "1.1"
    
    # Create meta information
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    
    # Basic task information
    task_id = ET.SubElement(task, "id")
    task_id.text = "1"
    
    name = ET.SubElement(task, "name")
    name.text = "COCO"
    
    size = ET.SubElement(task, "size")
    size.text = str(len(coco.getImgIds()))
    
    mode = ET.SubElement(task, "mode")
    mode.text = "annotation"
    
    overlap = ET.SubElement(task, "overlap")
    overlap.text = "0"
    
    bugtracker = ET.SubElement(task, "bugtracker")
    bugtracker.text = ""
    
    created = ET.SubElement(task, "created")
    created.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    
    updated = ET.SubElement(task, "updated")
    updated.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    
    subset = ET.SubElement(task, "subset")
    subset.text = "default"
    
    start_frame = ET.SubElement(task, "start_frame")
    start_frame.text = "0"
    
    stop_frame = ET.SubElement(task, "stop_frame")
    stop_frame.text = str(len(coco.getImgIds()) - 1)
    
    frame_filter = ET.SubElement(task, "frame_filter")
    frame_filter.text = ""
    
    # Segment information
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    
    seg_id = ET.SubElement(segment, "id")
    seg_id.text = "1"
    
    start = ET.SubElement(segment, "start")
    start.text = "0"
    
    stop = ET.SubElement(segment, "stop")
    stop.text = str(len(coco.getImgIds()) - 1)
    
    url = ET.SubElement(segment, "url")
    url.text = ""
    
    # Owner information
    owner = ET.SubElement(task, "owner")
    username = ET.SubElement(owner, "username")
    username.text = "user"
    
    email = ET.SubElement(owner, "email")
    email.text = "user@example.com"
    
    assignee = ET.SubElement(task, "assignee")
    assignee.text = ""
    
    # Label information
    labels = ET.SubElement(task, "labels")
    
    # Create labels from COCO categories
    categories = coco.loadCats(coco.getCatIds())
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", 
              "#800000", "#008000", "#000080", "#808000", "#800080", "#008080"]
    
    for i, category in enumerate(categories):
        label = ET.SubElement(labels, "label")
        
        label_name = ET.SubElement(label, "name")
        label_name.text = category['name']
        
        color = ET.SubElement(label, "color")
        color.text = colors[i % len(colors)]
        
        label_type = ET.SubElement(label, "type")
        label_type.text = "any"
        
        attributes = ET.SubElement(label, "attributes")
    
    # Process annotations
    img_ids = coco.getImgIds()
    
    for frame_id, img_id in enumerate(tqdm(img_ids, desc="Processing")):
        img_info = coco.loadImgs([img_id])[0]
        
        # Create image element
        image = ET.SubElement(annotations, "image")
        image.set("id", str(frame_id))
        image.set("name", img_info['file_name'])
        image.set("width", str(img_info['width']))
        image.set("height", str(img_info['height']))
        
        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # Get category information
            category = coco.loadCats([ann['category_id']])[0]
            
            # Process bounding box
            if 'bbox' in ann:
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                
                # Convert to CVAT format (xtl, ytl, xbr, ybr)
                xtl = bbox[0]
                ytl = bbox[1]
                xbr = bbox[0] + bbox[2]
                ybr = bbox[1] + bbox[3]
                
                # Create box element
                box = ET.SubElement(image, "box")
                box.set("label", category['name'])
                box.set("source", "file")
                box.set("occluded", "0")
                box.set("xtl", f"{xtl:.2f}")
                box.set("ytl", f"{ytl:.2f}")
                box.set("xbr", f"{xbr:.2f}")
                box.set("ybr", f"{ybr:.2f}")
                box.set("z_order", "0")
            
            # Process segmentation polygons
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    # Convert segmentation points to string format
                    points = []
                    for i in range(0, len(seg), 2):
                        if i + 1 < len(seg):
                            points.append(f"{seg[i]:.2f},{seg[i+1]:.2f}")
                    
                    if points:
                        # Create polygon element
                        polygon = ET.SubElement(image, "polygon")
                        polygon.set("label", category['name'])
                        polygon.set("source", "file")
                        polygon.set("occluded", "0")
                        polygon.set("points", ";".join(points))
                        polygon.set("z_order", "0")
    
    return annotations


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert COCO annotations to CVAT XML format")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="path to coco annotations JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="output path for CVAT XML file"
    )
    args = parser.parse_args()

    print(f"Loading COCO annotations from: {args.input_path}")

    # Read COCO annotations
    coco = COCO(args.input_path)
    
    print(f"Found {len(coco.getImgIds())} images, {len(coco.getCatIds())} categories")
    
    # Convert to CVAT format
    print("Converting to CVAT format...")
    cvat_xml = create_cvat_xml(coco)

    # Create output directory
    print(f"Saving CVAT XML to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Format XML
    def indent(elem, level=0):
        """Beautify XML format"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent(cvat_xml)
    tree = ET.ElementTree(cvat_xml)
    
    # Write file
    with open(args.output_path, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        tree.write(f, encoding='utf-8')
    
    print("Conversion completed successfully!")
