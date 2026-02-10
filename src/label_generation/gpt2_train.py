import argparse

import torch
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, interval):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.interval = interval

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
        with torch.no_grad():
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.9,
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n[step {state.global_step}] prompt: {prompt}\n{text}\n")
        model.train()
        return control


def tokenize_function(examples, tokenizer):
    # Tokenize text
    output = tokenizer(examples["text"])
    return output


def group_texts(examples, block_size):
    # Concatenate and split into fixed length chunks
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
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
        default=1,
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
        default=3,
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
        default=["[large, dog], [large, cat], [large, couch], [large, couch] ;",
                 "[large, person], [medium, skis] ;"],
        help="Prompts for text generation during training"
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=500,
        help="Generate text samples every X steps"
    )
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and preprocess
    raw_datasets = load_dataset(args.dataset_name_or_path, split="train")

    tokenized = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    lm_datasets = tokenized.map(
        lambda x: group_texts(x, args.block_size),
        batched=True
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
        report_to="none",
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
        callbacks=[ValidationCallback(tokenizer, args.validation_prompts, args.validation_interval)]
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
    