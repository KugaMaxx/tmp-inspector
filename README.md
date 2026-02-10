# AI-Driven YOLO Dataset Generator

An end-to-end generative AI pipeline for object detection dataset generation and training. Leverages GPT-2 for label generation, Qwen for synthetic image generation, and ultimately trains YOLO object detection models.

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd tmp-inspector

# Install dependencies
pip install -r requirements.txt

# Additional dependencies (if needed)
pip install ultralytics torch pillow pyyaml
```

### Usage Pipeline

#### 1. Train GPT-2 Model

```bash
python src/label_generation/gpt2_train.py \
    --dataset_path <your-dataset-path> \
    --output_dir ./models/gpt2-bbox \
    --num_train_epochs 10
```

**Parameters:**
- `--dataset_path`: Training data path
- `--output_dir`: Model output directory
- `--num_train_epochs`: Number of training epochs

#### 2. Generate Bounding Box Labels

```bash
python src/label_generation/gpt2_generate.py \
    --model_path ./models/gpt2-bbox \
    --output_dir ./generated_labels \
    --num_samples 1000
```

**Parameters:**
- `--model_path`: Path to the trained GPT-2 model
- `--output_dir`: Output directory for generated labels
- `--num_samples`: Number of samples to generate

#### 3. Synthesize Training Images

```bash
python src/image_synthesis/qwen_generate.py \
    --class_config config/qwen_class_tmp.csv \
    --qwen_model Qwen/Qwen-Image-Edit-2511 \
    --resolution 1024 \
    --output_dir ./generated_dataset
```

**Parameters:**
- `--class_config`: Class configuration file
- `--qwen_model`: Qwen model path or HuggingFace identifier
- `--resolution`: Image resolution
- `--output_dir`: Output dataset directory

#### 4. Train YOLO Model

```bash
python src/object_detection/yolo_train.py
```

Or directly modify the dataset configuration path in the script:

```python
# Edit yolo_train.py
train_results = model.train(
    data="./data/your-dataset/data.yaml",
    epochs=100,
    imgsz=1024,
    device="0"
)
```
