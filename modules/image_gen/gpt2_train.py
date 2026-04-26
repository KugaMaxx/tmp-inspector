import argparse
import importlib
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    
try:
    import wandb
except ImportError:
    wandb = None


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
        default=1000,
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
        default=100,
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
        "--validation_interval",
        type=int,
        default=500,
        help="Generate text samples every X steps"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Comma separated tracking backends: none, tensorboard, wandb"
    )
    
    return parser.parse_args()


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, interval, report_to, output_dir):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.interval = interval
        self.report_to = report_to
        self.output_dir = output_dir
        self.tb_writer = None
        self.img_sz = 512

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
        if "tensorboard" in self.report_to and SummaryWriter is not None:
            # Initialize SummaryWriter if not already done
            if self.tb_writer is None:
                self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "validation_images"))

            # Log each image with a unique tag
            for i, img in enumerate(images):
                self.tb_writer.add_image(
                    tag=f"validation/prompt_{i}",
                    img_tensor=np.array(img),
                    global_step=state.global_step,
                    dataformats="HWC",
                )
                
            self.tb_writer.flush()

        if "wandb" in self.report_to and wandb is not None:
            payload = {
                f"validation/prompt_{i}": wandb.Image(img, caption=f"step={state.global_step}, prompt_id={i}")
                for i, img in enumerate(images)
            }
            wandb.log(payload, step=state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return control

        # Initialize model and device
        model = kwargs.get("model")
        model.eval()
        device = next(model.parameters()).device
        
        # Generate text for each prompt and decode
        parsed_results = []
        with torch.no_grad():
            for prompt in self.prompts:
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
                print(f"\n[prompt]: {prompt}\n [response]: {response}\n")
                
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
            
            # Update
            images.append(img)

        self._log_images(images, state)
        model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.close()
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
    # Set TOKENIZERS_PARALLELISM to suppress huggingface tokenizers warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse arguments
    args = parse_args()

    args.report_to = [r.strip() for r in args.report_to.split(",") if r.strip()]
    if len(args.report_to) == 0 or args.report_to == ["none"]:
        args.report_to = "none"
    
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

    # Data collator for language modeling (causal language modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

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
        report_to=args.report_to,
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
                prompts=args.validation_prompts,
                interval=args.validation_interval,
                report_to=args.report_to,
                output_dir=args.output_dir
            )
        ]
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
    