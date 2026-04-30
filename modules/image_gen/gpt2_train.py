import re
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
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
    parser = argparse.ArgumentParser(description="Train a GPT-2 model to generate bounding box priors from text prompts.")
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
        help="Name or path of the dataset (HuggingFace dataset name or local path)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Size of text blocks for training sequences"
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
        default=1000,
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
            f"[{'small' if area < 0.004 else 'medium' if area < 0.036 else 'large'}, {c}]"
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

    # Prepare accelerator
    accelerator = Accelerator(log_with=args.report_to, project_dir=args.output_dir)
    accelerator.init_trackers(args.tracker_name)

    # Set logging
    logging_dir = Path(args.output_dir) / args.tracker_name
    logging_dir.mkdir(parents=True, exist_ok=True)
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

    # Load dataset and preprocess
    datasets = load_dataset(args.dataset_name_or_path, split="train")

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
    