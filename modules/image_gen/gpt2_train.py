import re
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, Dataset
from transformers import set_seed
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

try:
    import wandb
except ImportError:
    wandb = None


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 for bbox priors, augmenting target classes with aspect-ratio-matched COCO objects."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='gpt2-fire',
        help="Directory to save model and logs"
    )

    # Dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Name or path of the target dataset (HuggingFace dataset name or local path)"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split of the target dataset to train on."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Size of text blocks for training sequences"
    )

    # COCO augmentation
    parser.add_argument(
        "--coco_dataset_name_or_path",
        type=str,
        default=None,
        help=
        (
            "Path to the COCO dataset in HuggingFace (parquet) format, read for augmentation."
            "If is None, COCO augmentation is disabled and training proceeds on the target dataset alone."
        )
    )
    parser.add_argument(
        "--coco_dataset_split",
        type=str,
        default="train",
        help="Split of the COCO dataset to draw augmentation boxes from."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of most aspect-ratio-similar COCO classes to import per fire class."
    )
    parser.add_argument(
        "--ar_percentile_low",
        type=float,
        default=10.0,
        help="Lower percentile used to describe a class' aspect-ratio range."
    )
    parser.add_argument(
        "--ar_percentile_high",
        type=float,
        default=90.0,
        help="Upper percentile used to describe a class' aspect-ratio range."
    )
    parser.add_argument(
        "--min_class_samples",
        type=int,
        default=10,
        help="Minimum number of boxes required to estimate a class' aspect-ratio range."
    )
    parser.add_argument(
        "--max_match_distance",
        type=float,
        default=0.5,
        help=(
            "Discard COCO matches whose log-aspect-ratio distance exceeds this value "
            "(0 = no filtering). ~0.5 corresponds to roughly a 1.3x aspect-ratio mismatch; "
            "a fire class with no match below the threshold is left un-augmented."
        )
    )
    parser.add_argument(
        "--max_aug_samples",
        type=int,
        default=0,
        help="Maximum number of COCO scenes to import as augmentation (0 = no limit)."
    )

    # Training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )

    # Logging
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X update steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log training metrics every X steps"
    )

    # Validation
    parser.add_argument(
        "--validation_prompts",
        nargs="+",
        default=[],
        help="Prompts for text generation during training"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Generate text samples every X steps"
    )
    parser.add_argument(
        "--report_to",
        nargs="+",
        default=["tensorboard", "wandb"],
        help="Comma separated tracking backends: none, tensorboard, wandb"
    )

    # Accelerator
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="gpt2-train",
        help="Name for the accelerator tracker (e.g., for TensorBoard or WandB logging)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Aspect-ratio based COCO augmentation
# ---------------------------------------------------------------------------
def load_objects_dataset(dataset_name_or_path, split):
    """Load a dataset and keep only the columns needed for box statistics."""
    datasets = load_dataset(dataset_name_or_path, split=split)
    keep = [c for c in datasets.column_names if c in ("objects", "width", "height")]
    return datasets.select_columns(keep)


def collect_aspect_ratios(datasets):
    """Collect per-class lists of bounding-box aspect ratios (pixel width / height)."""
    aspect_ratios = defaultdict(list)
    for example in datasets:
        objs = example["objects"]
        img_w = float(example.get("width") or 1.0)
        img_h = float(example.get("height") or 1.0)

        for name, box in zip(objs["category_name"], objs["bbox"]):
            box_w = float(box[2])
            box_h = float(box[3])
            if box_w <= 0.0 or box_h <= 0.0:
                continue

            aspect_ratios[name].append((box_w * img_w) / (box_h * img_h))

    return aspect_ratios


def aspect_ratio_features(aspect_ratios, p_low, p_high, min_samples):
    """Summarize each class' aspect-ratio range as a feature vector in log-space."""
    features = {}
    ranges = {}
    for name, ratios in aspect_ratios.items():
        if len(ratios) < min_samples:
            continue

        log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
        median = float(np.median(log_ratios))
        low = float(np.percentile(log_ratios, p_low))
        high = float(np.percentile(log_ratios, p_high))

        features[name] = np.array([median, low, high], dtype=np.float64)
        ranges[name] = (float(np.exp(low)), float(np.exp(high)), int(len(ratios)))

    return features, ranges


def match_coco_classes(tar_features, coco_features, top_k, max_distance):
    """For each target class, find the top-k COCO classes with the closest AR range."""
    mapping = {}
    for tar_name, tar_feat in tar_features.items():
        distances = [
            (coco_name, float(np.linalg.norm(tar_feat - coco_feat)))
            for coco_name, coco_feat in coco_features.items()
        ]
        if max_distance and max_distance > 0.0:
            distances = [item for item in distances if item[1] <= max_distance]
        distances.sort(key=lambda item: item[1])
        mapping[tar_name] = distances[:top_k]

    return mapping


def build_coco_augmentation(coco_datasets, mapping, tar_features, max_aug_samples):
    """Import COCO scenes, relabeling matched boxes with the target class name."""
    # Reverse map: coco class -> target classes that selected it.
    coco_to_tar = defaultdict(list)
    for tar_name, matches in mapping.items():
        for coco_name, _ in matches:
            coco_to_tar[coco_name].append(tar_name)

    tar_median_log_ar = {name: feat[0] for name, feat in tar_features.items()}

    records = []
    for example in coco_datasets:
        objs = example["objects"]
        img_w = float(example.get("width") or 1.0)
        img_h = float(example.get("height") or 1.0)

        new_bboxes = []
        new_names = []
        for name, box in zip(objs["category_name"], objs["bbox"]):
            candidates = coco_to_tar.get(name)
            if not candidates:
                continue

            box_w = float(box[2])
            box_h = float(box[3])
            if box_w <= 0.0 or box_h <= 0.0:
                continue

            log_ar = float(np.log((box_w * img_w) / (box_h * img_h)))
            tar_name = min(candidates, key=lambda f: abs(tar_median_log_ar[f] - log_ar))

            new_bboxes.append([float(box[0]), float(box[1]), box_w, box_h])
            new_names.append(tar_name)

        if not new_bboxes:
            continue

        records.append({
            "objects": {
                "bbox": new_bboxes,
                "category": [-1] * len(new_bboxes),
                "category_name": new_names,
            }
        })

        if max_aug_samples and len(records) >= max_aug_samples:
            break

    return records


def build_training_dataset(args):
    """Build the combined (fire + aspect-ratio-matched COCO) training dataset."""
    # Load the target fire dataset (objects only).
    tar_datasets = load_objects_dataset(args.dataset_name_or_path, args.dataset_split)
    tar_records = [{"objects": example["objects"]} for example in tar_datasets]
    logger.info(f"Loaded {len(tar_records)} samples from: {args.dataset_name_or_path}")

    if args.coco_dataset_name_or_path is None:
        logger.info("COCO augmentation disabled; training on original dataset only.")
        return Dataset.from_list(tar_records)

    # Compute aspect-ratio ranges for the fire classes.
    tar_ar = collect_aspect_ratios(tar_datasets)
    tar_features, tar_ranges = aspect_ratio_features(
        tar_ar, args.ar_percentile_low, args.ar_percentile_high, args.min_class_samples
    )

    logger.info("Dataset class aspect-ratio ranges (w/h):")
    for name, (low, high, count) in sorted(tar_ranges.items()):
        logger.info(f"  {name:<28} [{low:.2f}, {high:.2f}]  (n={count})")

    # Compute aspect-ratio ranges for the COCO classes.
    coco_datasets = load_objects_dataset(args.coco_dataset_name_or_path, args.coco_dataset_split)
    coco_ar = collect_aspect_ratios(coco_datasets)
    coco_features, _ = aspect_ratio_features(
        coco_ar, args.ar_percentile_low, args.ar_percentile_high, args.min_class_samples
    )
    logger.info(f"Computed aspect-ratio ranges for {len(coco_features)} COCO classes.")

    # Match each fire class to its top-k aspect-ratio-similar COCO classes,
    # dropping matches whose aspect-ratio range is too dissimilar.
    mapping = match_coco_classes(tar_features, coco_features, args.top_k, args.max_match_distance)
    logger.info(
        f"Top-{args.top_k} aspect-ratio-matched COCO classes per fire class "
        f"(max_match_distance={args.max_match_distance}):"
    )
    for tar_name, matches in mapping.items():
        if matches:
            pretty = ", ".join(f"{coco_name} (d={dist:.3f})" for coco_name, dist in matches)
        else:
            pretty = "(no match below threshold - left un-augmented)"
        logger.info(f"  {tar_name:<28} <- {pretty}")

    # Import the matched COCO scenes as augmentation.
    aug_records = build_coco_augmentation(
        coco_datasets, mapping, tar_features, args.max_aug_samples
    )
    logger.info(f"Imported {len(aug_records)} COCO scenes as augmentation.")

    combined = tar_records + aug_records
    datasets = Dataset.from_list(combined).shuffle(seed=args.seed)
    logger.info(f"Combined training dataset size: {len(datasets)} samples.")

    return datasets


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, accelerator, logging_steps, validation_prompts, validation_steps, img_sz=512):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.logging_steps = logging_steps
        self.validation_prompts = validation_prompts
        self.validation_steps = validation_steps
        self.img_sz = img_sz

    @staticmethod
    def _parse_string(decoded_text):
        if ";" not in decoded_text:
            return None

        # extract prompt and response parts
        prompt, response = decoded_text.split(";", 1)
        response = response.split(";", 1)[0]

        # extract info
        labels = re.findall(r"\[(.*?)\]", prompt)
        bboxes = re.findall(r"\[(.*?)\]", response)

        # check if counts match and are non-empty
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

    def _log_images(self, images, state):
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard" and tracker.writer is not None:
                for i, img in enumerate(images):
                    tracker.writer.add_image(
                        tag=f"validation/prompt_{i}",
                        img_tensor=np.array(img),
                        global_step=state.global_step,
                        dataformats="HWC",
                    )
                tracker.writer.flush()

            elif tracker.name == "wandb" and wandb is not None:
                tracker.log({
                    f"validation/prompt_{i}": wandb.Image(img, caption=f"step={state.global_step}, prompt_id={i}")
                    for i, img in enumerate(images)
                }, step=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        if self.logging_steps <= 0 or state.global_step == 0 or state.global_step % self.logging_steps != 0:
            return control
        
        logger.info(
            f"step: {state.global_step} | "
            f"loss: {logs.get('loss', 'N/A'):.4f} | "
            f"grad_norm: {logs.get('grad_norm', 'N/A'):.4f} | "
            f"learning_rate: {logs.get('learning_rate', 'N/A'):.4e}"
        )

        scalar = {
            "train/epoch": state.epoch,
            "train/loss": logs.get("loss"),
            "train/grad_norm": logs.get("grad_norm"),
            "train/learning_rate": logs.get("learning_rate"),
        }
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard" and tracker.writer is not None:
                for key, value in scalar.items():
                    tracker.writer.add_scalar(key, value, state.global_step)
                tracker.writer.flush()

            elif tracker.name == "wandb" and wandb is not None:
                tracker.log(scalar, step=state.global_step)

        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.validation_steps <= 0:
            return control
        
        if state.global_step == 0 or state.global_step % self.validation_steps != 0:
            return control

        if not self.accelerator.is_main_process:
            return control

        # Initialize model and device
        model = kwargs.get("model")
        model.eval()
        device = next(model.parameters()).device

        # Generate text for each prompt and decode
        parsed_results = []
        with torch.no_grad():
            for prompt in self.validation_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Log the prompt and response for debugging
                logger.info(f"PROMPT:   {prompt}")
                logger.info(f"RESPONSE: {response}")

                # Attempt to parse the response and log the results
                parsed_results.append(self._parse_string(response))

        images = []
        for result in parsed_results:
            # Create a blank image
            img = Image.new("RGB", (self.img_sz, self.img_sz), "black")
            draw = ImageDraw.Draw(img)

            # If parsing failed, log a default image indicating invalid output
            if result is None:
                draw.text((10, 10), "Invalid output", fill="white")

            # If parsing succeeded, render the bounding boxes on an image
            else:
                for label, (x, y, w, h) in result:
                    x1 = int((x - w / 2.0) * self.img_sz)
                    y1 = int((y - h / 2.0) * self.img_sz)
                    x2 = int((x + w / 2.0) * self.img_sz)
                    y2 = int((y + h / 2.0) * self.img_sz)

                    x1 = max(0, min(self.img_sz - 1, x1))
                    y1 = max(0, min(self.img_sz - 1, y1))
                    x2 = max(0, min(self.img_sz - 1, x2))
                    y2 = max(0, min(self.img_sz - 1, y2))

                    draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=2)
                    draw.text((x1, max(0, y1 - 12)), label, fill="white")
            
            # Update images list
            images.append(img)

        self._log_images(images, state)
        model.train()

        return control


def preprocess(examples, tokenizer, block_size):
    # Construct training data
    texts = []
    for objs in examples["objects"]:
        
        # Create prompt with size category and class name
        prompt = [
            f"[{'xs' if area < 1/256 else 's' if area < 1/64 else 'm' if area < 1/16 else 'l' if area < 1/4 else 'xl'}, {c}]"
            for c, b in zip(objs["category_name"], objs["bbox"])
            for area in (float(b[2]) * float(b[3]),)
        ]

        # Create response with normalized bbox coordinates
        response = [
            f"[{', '.join(f'{float(v):.3f}' for v in b)}]"
            for b in objs["bbox"]
        ]

        # Join them together and append the EOS token
        text = " , ".join(prompt) + " ; " + " , ".join(response) + " ; "
        texts.append(text)

    # Tokenize texts
    tokenized = tokenizer(texts)

    # Concatenate and split into fixed length chunks
    concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()

    return result


def main():
    # Parse arguments
    args = parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Define logging directory
    logging_dir = Path(args.output_dir) / "logs" / args.tracker_name
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Prepare accelerator
    accelerator = Accelerator(log_with=args.report_to, project_dir=Path(args.output_dir) / "logs")
    accelerator.init_trackers(
        args.tracker_name,
        init_kwargs={
            "wandb": {"dir": logging_dir} if wandb is not None else {},
        }
    )

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(logging_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ],
    )
    logger.info(f"Starting script: {Path(__file__).name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset and preprocess
    datasets = build_training_dataset(args)

    lm_datasets = datasets.map(
        lambda x: preprocess(x, tokenizer, args.block_size),
        batched=True,
        remove_columns=datasets.column_names,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)

    # Logging configuration and model details
    logger.info(f"Training Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"Model Config: \n {model.config}")

    # Data collator for language modeling (causal language modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Check if checkpointing steps and validation steps are set, otherwise use default values
    steps_per_epoch = len(lm_datasets) // args.per_device_train_batch_size
    args.save_steps = args.save_steps if args.save_steps else steps_per_epoch
    args.logging_steps = args.logging_steps if args.logging_steps else steps_per_epoch
    args.validation_steps = args.validation_steps if args.validation_steps else steps_per_epoch

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
        callbacks=[
            ValidationCallback(
                tokenizer=tokenizer,
                accelerator=accelerator,
                logging_steps=args.logging_steps,
                validation_prompts=args.validation_prompts,
                validation_steps=args.validation_steps,
            )
        ]
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # End of training
    accelerator.end_training()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
