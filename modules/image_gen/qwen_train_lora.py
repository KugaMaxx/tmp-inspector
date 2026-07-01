import math
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import set_seed
from datasets import load_dataset

from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.optimization import get_scheduler
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLIGEN-style LoRA training for Qwen-Image-Edit-2511."
    )

    # Qwen model
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Path to Qwen image editing model or model id on huggingface."
    )
    parser.add_argument(
        "--qwen_prompt",
        type=str,
        default=(
            "Replace the black background with a coherent real-world scene. "
            "Place each object only inside its colored mask and make it fill that mask; "
            "do not add these objects anywhere outside their mask."
        ),
        help="Global scene instruction prepended once before the per-bbox object prompts.",
    )
    parser.add_argument(
        "--qwen_grounded_prompt",
        type=str,
        default=(
            "Replace the {color} mask with {objects}."
        ),
        help="Per-bbox template for the object placed in each masked region.",
    )
    parser.add_argument(
        "--qwen_negative_prompt",
        type=str,
        default=(
            "watermark, object outside mask"
        ),
        help="Negative prompt for image generation.",
    )

    # Dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Name or path of the target dataset (HuggingFace dataset name or local path)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to use from the dataset (0 for all samples)."
    )

    # LoRA
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of the LoRA adapters."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=0,
        help="Scaling factor for the LoRA adapters. If 0, it will be set to the same value as --lora_rank."
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0",
                  "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        help="List of target modules in the Qwen model to apply LoRA adapters."
    )

    # Directories       
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save LoRA adapters and training logs."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory to save training logs (TensorBoard, WandB, etc.).",
    )

    # Training
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int, 
        default=4, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int, 
        default=3,
        help="Total number of training epochs to perform. Ignored if max_train_steps is provided."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Scheduler
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay used by AdamW.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print training logs every N optimizer steps.",
    )

    # Sampler
    parser.add_argument(
        "--weighting_scheme", 
        type=str, 
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Timestep density / loss weighting scheme for flow-matching."
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # Validation
    parser.add_argument(
        "--validation_ids",
        type=int,
        nargs="*",
        default=None,
        help=("A set of validation data evaluated every `--validation_steps`."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=40,
        help="Denoising steps used for validation image generation.",
    )
    parser.add_argument(
        "--validation_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale used for validation image generation.",
    )
    parser.add_argument(
        "--report_to",
        nargs="+",
        default=["tensorboard", "wandb"],
        help="Comma separated tracking backends: none, tensorboard, wandb"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. 0 disables checkpointing. "
            "These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`. "
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. "
            "None means no checkpoints, 0 means keep all. "
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Dataloader
# ------------------------------------------------------------------------------

COLOR_PALETTE = [
    ("red",     (255,   0,   0)),
    ("green",   (  0, 200,   0)),
    ("blue",    (  0,   0, 255)),
    ("yellow",  (255, 255,   0)),
    ("magenta", (255,   0, 255)),
    ("cyan",    (  0, 255, 255)),
    ("orange",  (255, 128,   0)),
    ("purple",  (128,   0, 128)),
    ("pink",    (255, 192, 203)),
    ("lime",    (  0, 255,   0)),
    ("teal",    (  0, 128, 128)),
    ("brown",   (165,  42,  42)),
]


def allocate_bbox_colors(n):
    palette = COLOR_PALETTE.copy()
    random.shuffle(palette)
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def draw_condition_image(bboxes, width, height, colors, alpha=128, draw_labels=False, background=None):
    if background is None:
        base = Image.new("RGBA", (width, height), color=(0, 0, 0, 255))
    else:
        base = background.resize((width, height)).convert("RGBA")
    
    overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=24)

    for (label, bbox), (_, rgb) in zip(bboxes, colors):
        x, y, w, h = bbox
        left   = int((x - w / 2.0) * width)
        top    = int((y - h / 2.0) * height)
        right  = int((x + w / 2.0) * width)
        bottom = int((y + h / 2.0) * height)

        # semi-transparent mask fill
        draw.rectangle([left, top, right, bottom], fill=(*rgb, alpha))

        if draw_labels and label is not None:
            (text_width, text_height), (_, _) = font.font.getsize(label)
            text_x = left if left + text_width < width else width - text_width
            text_y = top if top + text_height < height else height - text_height
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


def prepare_dataset(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name_or_path, cache_dir=args.cache_dir)

    # Drop rows without boxes
    dataset = dataset.filter(lambda obj: len(obj["bbox"]) > 0, input_columns="objects")

    # Randomly sample a subset if requested
    if args.max_samples and args.max_samples > 0:
        dataset['train'] = dataset['train'].select(range(min(args.max_samples, len(dataset['train']))))

    # Extract a subset for efficient validation
    if args.validation_ids is not None:
        dataset["validation"] = dataset["validation"].select(args.validation_ids)

    def process(example):
        bboxes = example["objects"].get("bbox")
        labels = example["objects"].get("category_name")
        bboxes = [(label, bbox) for label, bbox in zip(labels, bboxes)]
        colors = allocate_bbox_colors(len(bboxes))

        target_image = example["image"].convert("RGB")
        tw, th = target_image.size
        ratio = tw / th

        # Build condition image
        vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, ratio)
        cond_image = draw_condition_image(bboxes, vae_w, vae_h, colors, alpha=255)

        # Build qwen prompt
        qwen_grounded_prompt = " ".join(
            args.qwen_grounded_prompt.format(
                objects=label,
                color=color,
            )
            for (label, bbox), (color, rgb) in zip(bboxes, colors)
        )
        qwen_prompt = f"{args.qwen_prompt} {qwen_grounded_prompt}"
        qwen_negative_prompt = args.qwen_negative_prompt

        return {
            "bboxes": bboxes,
            "colors": colors,
            "prompt": qwen_prompt,
            "negative_prompt": qwen_negative_prompt,
            "cond_image": cond_image,
            "target_image": target_image,
        }

    def collate_fn(examples):
        # DataLoader hands collate_fn a list of dataset rows (the batch).
        return [process(example) for example in examples]

    dataloader = {}
    for dataset_part in dataset.keys():
        dataloader[dataset_part] = torch.utils.data.DataLoader(
            dataset[dataset_part],
            shuffle=True if dataset_part == 'train' else False,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size if dataset_part == 'train' else 1,
            num_workers=args.dataloader_num_workers,
        )

    return dataloader


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

# Constants copied from QwenImageEditPlusPipeline so encoding matches inference exactly.
CONDITION_IMAGE_SIZE = 384 * 384      # area used for the text-encoder vision tokens
VAE_IMAGE_SIZE = 1024 * 1024          # area used for the VAE latents / output resolution
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 64
IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _encode_vae_image(vae, image, latent_channels):
    image_latents = vae.encode(image).latent_dist.mode()  # deterministic (sample_mode="argmax")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    return image_latents


def _extract_masked_hidden(hidden_states, mask):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def _get_qwen_prompt_embeds(text_encoder, processor, prompt, condition_images, device, dtype):
    base_img_prompt = ""
    for i in range(len(condition_images)):
        base_img_prompt += IMG_PROMPT_TEMPLATE.format(i + 1)

    txt = [PROMPT_TEMPLATE_ENCODE.format(base_img_prompt + prompt)]
    model_inputs = processor(
        text=txt,
        images=condition_images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        pixel_values=model_inputs.get("pixel_values"),
        image_grid_thw=model_inputs.get("image_grid_thw"),
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split = _extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
    split = [e[PROMPT_TEMPLATE_ENCODE_START_IDX:] for e in split]

    # single sample -> no padding needed
    prompt_embeds = split[0].unsqueeze(0).to(dtype=dtype, device=device)
    prompt_embeds_mask = torch.ones(
        (1, prompt_embeds.shape[1]), dtype=torch.long, device=device
    )

    return prompt_embeds, prompt_embeds_mask


@torch.no_grad()
def encode_sample(sample, vae, text_encoder, processor, image_processor, latent_channels, device, dtype):
    # ----- prompt embeddings -----
    cond_image = sample["cond_image"]
    vae_w, vae_h = cond_image.width, cond_image.height
    cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, vae_w / vae_h)
    cond_image_resized = image_processor.resize(cond_image, cond_h, cond_w)
    prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(
        text_encoder, processor, sample["prompt"], [cond_image_resized], device, dtype
    )

    # ----- target latents -----
    target_image = sample["target_image"]
    target_tensor = image_processor.preprocess(target_image, vae_h, vae_w).unsqueeze(2)
    target_lat = _encode_vae_image(
        vae, target_tensor.to(device, dtype), latent_channels
    )
    lh, lw = target_lat.shape[3], target_lat.shape[4]
    target_latents = _pack_latents(target_lat, 1, latent_channels, lh, lw)

    # ----- condition latents -----
    cond_tensor = image_processor.preprocess(cond_image, vae_h, vae_w).unsqueeze(2)
    cond_lat = _encode_vae_image(
        vae, cond_tensor.to(device, dtype), latent_channels
    )
    clh, clw = cond_lat.shape[3], cond_lat.shape[4]
    cond_latents = _pack_latents(cond_lat, 1, latent_channels, clh, clw)

    # img_shapes: (frames, grid_h, grid_w) for target then condition, wrapped for batch=1
    img_shapes = [[(1, lh // 2, lw // 2), (1, clh // 2, clw // 2)]]

    return {
        "prompt_embeds": prompt_embeds,            # (1, L, D)
        "prompt_embeds_mask": prompt_embeds_mask,  # (1, L)
        "target_latents": target_latents,          # (1, S, 64)
        "cond_latents": cond_latents,              # (1, Sc, 64)
        "img_shapes": img_shapes,
    }


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def init_trackers(args):
    trackers = {}
    report_to = [b for b in (args.report_to or []) if b and b != "none"]

    if "tensorboard" in report_to:
        try:
            from torch.utils.tensorboard import SummaryWriter
            trackers["tensorboard"] = SummaryWriter(log_dir=str(Path(args.output_dir) / "logs"))
        except ImportError:
            logger.warning("[validation] tensorboard requested but not installed; skipping.")

    if "wandb" in report_to:
        try:
            import wandb
            wandb.init(
                project="qwen-gligen-lora",
                dir=str(args.output_dir),
                config=vars(args),
            )
            trackers["wandb"] = wandb
        except ImportError:
            logger.warning("[validation] wandb requested but not installed; skipping.")

    return trackers


@torch.no_grad()
def log_validation(args, pipeline, trackers, dataloader, global_step, device):
    if not trackers:
        return

    logger.info(f"[validation] running validation at step {global_step}...")

    transformer = pipeline.transformer
    was_training = transformer.training
    transformer.eval()
    pipeline.set_progress_bar_config(disable=True)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    images, captions = [], []
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    for i, sample in enumerate(dataloader["validation"]):
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            image = pipeline(
                image=[sample[0]["cond_image"]],
                prompt=sample[0]["prompt"],
                negative_prompt=sample[0]["negative_prompt"],
                num_inference_steps=args.validation_num_inference_steps,
                true_cfg_scale=args.validation_cfg_scale,
                generator=torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None,
            ).images[0]

        # Combine the condition image and generated image
        cond_image_w_label = draw_condition_image(
            bboxes=sample[0]["bboxes"],
            width=sample[0]["cond_image"].width,
            height=sample[0]["cond_image"].height,
            colors=sample[0]["colors"],
            alpha=64,
            draw_labels=True,
            background=image
        )

        images.append(cond_image_w_label)
        captions.append(sample[0]["prompt"])
    
    if "tensorboard" in trackers:
        import numpy as np
        writer = trackers["tensorboard"]
        for i, image in enumerate(images):
            arr = np.asarray(image.convert("RGB"))  # (H, W, C)
            writer.add_image(f"validation/sample_{i}", arr, global_step, dataformats="HWC")
    
    if "wandb" in trackers:
        wandb = trackers["wandb"]
        wandb.log(
            {
                "validation": [
                    wandb.Image(image, caption=caption)
                    for image, caption in zip(images, captions)
                ]
            },
            step=global_step,
        )

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if was_training:
        transformer.train()


# ------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------

def save_checkpoint(args, transformer, optimizer, lr_scheduler, global_step, epoch):
    if args.checkpoints_total_limit is None:
        return

    # Create checkpoint directory
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    lora_state = get_peft_model_state_dict(transformer)
    QwenImageLoraLoaderMixin.save_lora_weights(
        save_directory=str(ckpt_dir),
        transformer_lora_layers=lora_state,
        safe_serialization=True,
    )

    # Resumable training state.
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "lora_state_dict": get_peft_model_state_dict(transformer),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        ckpt_dir / "training_state.pt",
    )
    logger.info(f"[checkpoint] saved state to {ckpt_dir}")

    _prune_checkpoints(args)
    return ckpt_dir


def _prune_checkpoints(args):
    if args.checkpoints_total_limit <= 0:
        return

    checkpoints = sorted(
        Path(args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if len(checkpoints) <= args.checkpoints_total_limit:
        return

    import shutil
    to_remove = checkpoints[: len(checkpoints) - args.checkpoints_total_limit]
    for ckpt in to_remove:
        shutil.rmtree(ckpt, ignore_errors=True)
    logger.info(f"[checkpoint] pruned {len(to_remove)} old checkpoint(s): "
                f"{', '.join(p.name for p in to_remove)}")


def load_checkpoint(args, transformer, optimizer, lr_scheduler):
    ckpt_dir = _resolve_checkpoint_path(args)
    if ckpt_dir is None:
        return 0, 0

    state_path = ckpt_dir / "training_state.pt"
    if not state_path.exists():
        logger.info(f"[checkpoint] {state_path} not found; starting from scratch.")
        return 0, 0

    logger.info(f"[checkpoint] resuming from {ckpt_dir}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    # LoRA weights -> peft adapters (must already be added on `transformer`)
    set_peft_model_state_dict(transformer, state["lora_state_dict"])

    # Keep LoRA params in fp32, matching the fresh-init path
    cast_training_params([transformer], dtype=torch.float32)

    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])

    global_step = int(state["global_step"])
    epoch = int(state["epoch"])
    logger.info(f"[checkpoint] resumed at step {global_step} (epoch {epoch})")
    return global_step, epoch


def _resolve_checkpoint_path(args):
    resume = args.resume_from_checkpoint
    if resume is None:
        return None

    if resume == "latest":
        checkpoints = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            logger.info("[checkpoint] --resume_from_checkpoint=latest but no checkpoint found; "
                        "starting from scratch.")
            return None
        return checkpoints[-1]

    path = Path(resume)
    if not path.exists():
        logger.info(f"[checkpoint] --resume_from_checkpoint={resume} does not exist; "
                    "starting from scratch.")
        return None
    return path


# ------------------------------------------------------------------------------
# Others
# ------------------------------------------------------------------------------

def get_sigmas(scheduler, timesteps, device, n_dim, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def main():
    # Parser arguments
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16

    # Set random seeds for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set logging
    logging_dir = Path(args.output_dir) / args.logging_dir
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
    logger.info(f"[train] Starting script: {Path(__file__).name}")

    # Load Qwen model and tokenizer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.qwen_model, subfolder="transformer", torch_dtype=dtype, cache_dir=args.cache_dir
    )
    transformer.to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    # Load VAE + text encoder + processor.
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.qwen_model, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    vae.requires_grad_(False)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    text_encoder.requires_grad_(False)
    processor = Qwen2VLProcessor.from_pretrained(args.qwen_model, subfolder="processor")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.qwen_model, subfolder="tokenizer")

    latent_channels = vae.config.z_dim
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    # Logging configuration and model details
    logger.info(f"[train] Training Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"[train] Model Config: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(transformer.config).items()])} \n")

    # Build dataset
    dataloader = prepare_dataset(args)

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.qwen_model, subfolder="scheduler"
    )
    num_train_timesteps = scheduler.config.num_train_timesteps

    # Load LoRA adapters
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha > 0 else args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=args.lora_target_modules,
    )
    transformer.add_adapter(lora_config)

    # Keep LoRA params in fp32 for stable optimization
    cast_training_params([transformer], dtype=torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    # Preprocess scheduler
    num_steps_per_epoch = len(dataloader["train"]) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = max(1, args.num_train_epochs * num_steps_per_epoch)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay
    )

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Setup training loop
    transformer.train()
    optimizer.zero_grad(set_to_none=True)

    # Check if checkpointing steps and validation steps are set, otherwise use default values
    args.validation_steps    = args.validation_steps if args.validation_steps else num_steps_per_epoch
    args.checkpointing_steps = args.checkpointing_steps if args.checkpointing_steps else num_steps_per_epoch

    # Resume from checkpoint if requested
    global_step, first_epoch = load_checkpoint(args, transformer, optimizer, lr_scheduler)
    accum = 0  # gradient-accumulation counter; checkpoints land on update boundaries, so this resets to 0

    # When resuming mid-epoch, skip the optimizer steps already completed in `first_epoch`
    completed_steps_in_epoch = global_step - first_epoch * num_steps_per_epoch
    resume_step = completed_steps_in_epoch * args.gradient_accumulation_steps

    # Setup validation trackers
    trackers = init_trackers(args)
    pipeline = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler.config),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )

    # Initialize progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
    )

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"[train] Starting epoch {epoch + 1}/{args.num_train_epochs}...")

        for step, batch in enumerate(dataloader["train"]):
            if global_step >= args.max_train_steps:
                break

            # Skip batches already processed before the resume point (only in the first epoch).
            if epoch == first_epoch and step < resume_step:
                continue

            # Encode the batch of samples
            samples = [
                encode_sample(
                    sample,
                    vae=vae,
                    text_encoder=text_encoder,
                    processor=processor,
                    image_processor=image_processor,
                    latent_channels=latent_channels,
                    device=device,
                    dtype=dtype,
                )
                for sample in batch
            ]
            
            # Concatenate for the batch
            target_latents = torch.cat([s["target_latents"] for s in samples], dim=0)   # (B,S,64)
            cond_latents   = torch.cat([s["cond_latents"] for s in samples],   dim=0)   # (B,Sc,64)

            bsz = len(samples)
            max_len = max(s["prompt_embeds"].shape[1] for s in samples)
            embed_dim = samples[0]["prompt_embeds"].shape[-1]
            prompt_embeds = torch.zeros((bsz, max_len, embed_dim), dtype=dtype, device=device)
            prompt_embeds_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
            for i, s in enumerate(samples):
                length = s["prompt_embeds"].shape[1]
                prompt_embeds[i, :length] = s["prompt_embeds"][0]
                prompt_embeds_mask[i, :length] = s["prompt_embeds_mask"][0]

            # One layer-list per batch element; all identical given the fixed image size.
            img_shapes = [s["img_shapes"][0] for s in samples]

            # Sample noise for flow-matching target
            noise = torch.randn_like(target_latents)

            # Sample a timestep per the chosen density scheme (one per batch element)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
            timesteps = scheduler.timesteps[indices].to(device)

            sigmas = get_sigmas(scheduler, timesteps, device, target_latents.ndim, dtype)
            noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

            # Condition latents are clean and concatenated along the sequence dimension
            model_input = torch.cat([noisy_latents, cond_latents], dim=1)

            # Compute guidance if the model has guidance embeddings
            guidance = None
            if bool(transformer.config.guidance_embeds):
                guidance = torch.full([bsz], 1.0, device=device, dtype=torch.float32)

            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with torch.autocast(device_type=autocast_device, dtype=dtype, enabled=device.startswith("cuda")):
                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
            model_pred = model_pred[:, : target_latents.size(1)]

            # flow-matching target = noise - x0; loss in fp32
            target_v = (noise - target_latents).float()
            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas).float()
            loss = (weighting * (model_pred.float() - target_v) ** 2).mean()

            (loss / args.gradient_accumulation_steps).backward()
            accum += 1

            if accum == args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_every == 0:
                    logger.info(
                        f"[train] step {global_step}/{args.max_train_steps} | "
                        f"loss {loss.item():.4f} lr {lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        args,
                        transformer=transformer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        epoch=epoch,
                    )

                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    log_validation(
                        args,
                        pipeline=pipeline,
                        trackers=trackers,
                        dataloader=dataloader,
                        global_step=global_step,
                        device=device,
                    )

                if global_step >= args.max_train_steps:
                    break

    # Save final LoRA weights
    lora_state = get_peft_model_state_dict(transformer)
    QwenImageLoraLoaderMixin.save_lora_weights(
        save_directory=str(output_dir),
        transformer_lora_layers=lora_state,
        safe_serialization=True,
    )

    # Run a final validation
    log_validation(
        args,
        pipeline=pipeline,
        trackers=trackers,
        dataloader=dataloader,
        global_step=global_step,
        device=device,
    )

    # Flush / close logging backends
    if "tensorboard" in trackers:
        trackers["tensorboard"].close()

    if "wandb" in trackers:
        trackers["wandb"].finish()

    logger.info(f"[train] finished training at step {global_step}. LoRA weights saved to {output_dir}")


if __name__ == "__main__":
    main()
