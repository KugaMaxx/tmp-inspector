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


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, interval, report_to, output_dir):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.interval = interval
        self.report_to = report_to
        self.output_dir = output_dir
        self.tb_writer = None

    @staticmethod
    def _extract_labels_from_prompt(prompt):
        prompt_part = prompt.split(";", 1)[0]
        labels = []
        for item in re.findall(r"\[(.*?)\]", prompt_part):
            parts = [p.strip() for p in item.split(",", 1)]
            if len(parts) == 2:
                labels.append(parts[1])
        return labels

    @staticmethod
    def _parse_bbox_response(decoded_text):
        if ";" not in decoded_text:
            return None

        response_part = decoded_text.split(";", 1)[1]
        raw_boxes = re.findall(r"\[(.*?)\]", response_part)
        if len(raw_boxes) == 0:
            return None

        bboxes = []
        for raw in raw_boxes:
            vals = [v.strip() for v in raw.split(",")]
            if len(vals) != 4:
                return None
            try:
                x, y, w, h = [float(v) for v in vals]
            except ValueError:
                return None

            # Strict [x, y, w, h] normalized format
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                return None
            bboxes.append([x, y, w, h])

        return bboxes

    @staticmethod
    def _default_bw_image(size=512):
        img = Image.new("RGB", (size, size), "black")
        draw = ImageDraw.Draw(img)
        draw.line((0, 0, size - 1, size - 1), fill="white", width=2)
        draw.line((0, size - 1, size - 1, 0), fill="white", width=2)
        return img

    @staticmethod
    def _render_bboxes(prompt, bboxes, size=512):
        labels = ValidationCallback._extract_labels_from_prompt(prompt)
        n = min(len(labels), len(bboxes))

        img = Image.new("RGB", (size, size), "black")
        draw = ImageDraw.Draw(img)
        for label, (x, y, w, h) in zip(labels[:n], bboxes[:n]):
            x1 = int(x * size)
            y1 = int(y * size)
            x2 = x1 + int(w * size)
            y2 = y1 + int(h * size)

            x1 = max(0, min(size - 1, x1))
            y1 = max(0, min(size - 1, y1))
            x2 = max(0, min(size - 1, x2))
            y2 = max(0, min(size - 1, y2))

            draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=2)
            draw.text((x1, max(0, y1 - 12)), label, fill="white")

        return img

    def _log_images(self, images, state):
        if "tensorboard" in self.report_to and SummaryWriter is not None:
            if self.tb_writer is None:
                self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "validation_images"))
            for i, img in enumerate(images):
                self.tb_writer.add_image(
                    tag=f"validation/prompt_{i}",
                    img_tensor=np.array(img),
                    global_step=state.global_step,
                    dataformats="HWC",
                )
                img.save(f'{i}.png')
                
            self.tb_writer.flush()

        if "wandb" in self.report_to:
            try:
                wandb_module = importlib.import_module("wandb")
            except ImportError:
                wandb_module = None

            if wandb_module is not None and wandb_module.run is not None:
                payload = {
                    f"validation/prompt_{i}": wandb_module.Image(img, caption=f"step={state.global_step}, prompt_id={i}")
                    for i, img in enumerate(images)
                }
                wandb_module.log(payload, step=state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        device = next(model.parameters()).device
        model.eval()
        decoded_results = []
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
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                decoded_results.append((prompt, text))
                print(f"\n[step {state.global_step}] prompt: {prompt}\n{text}\n")

        # Special case 1: if any generated response after ';' is invalid, all prompts use default B/W image.
        invalid_exists = False
        parsed_bboxes = []
        for _, text in decoded_results:
            bboxes = self._parse_bbox_response(text)
            if bboxes is None:
                invalid_exists = True
            parsed_bboxes.append(bboxes)

        if invalid_exists:
            images = [self._default_bw_image(512) for _ in self.prompts]
        else:
            # Special case 2: if bbox count and prompt-object count mismatch,
            # only render min(prompt_count, bbox_count) in order.
            images = [
                self._render_bboxes(prompt, bboxes, size=512)
                for (prompt, _), bboxes in zip(decoded_results, parsed_bboxes)
            ]

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
        text = " , ".join(prompt) + " ; " + " , ".join(response) + tokenizer.eos_token
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
    
    parser = argparse.ArgumentParser(description="Train a GPT-2 model to generate bounding box priors from text prompts.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="openai-community/gpt2",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/tmp-gpt2-fire-w-val",
        help="Directory to save model and logs"
    )

    # Dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/data/tmp-fire",
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
        default=200,
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
        default=[
            "[large, fire exit sign], [large, fire hose reel], [medium, fire alarm] ;",
            "[large, fire extinguisher], [medium, fire exit sign] ;"
        ],
        help="Prompts for text generation during training"
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="Generate text samples every X steps"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Comma separated tracking backends: none, tensorboard, wandb"
    )
    args = parser.parse_args()

    report_to = [r.strip() for r in args.report_to.split(",") if r.strip()]
    if len(report_to) == 0 or report_to == ["none"]:
        report_to = "none"
    
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
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        report_to=report_to,
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
        callbacks=[ValidationCallback(tokenizer, args.validation_prompts, args.validation_interval, report_to, args.output_dir)]
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
    