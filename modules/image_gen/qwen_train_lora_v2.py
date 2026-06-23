import math
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from PIL import Image, ImageDraw

import torch
from transformers import set_seed
from datasets import load_dataset

from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
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
from peft import LoraConfig, get_peft_model_state_dict


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
            "Replace the {color} masked region with {objects}."
        ),
        help="Per-bbox template for image generation prompt.",
    )
    parser.add_argument(
        "--qwen_negative_prompt",
        type=str,
        default=(
            " "
        ),
        help="Negative prompt for image generation.",
    )

    # Dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/data/Fire-Art/hf_datasets",
        help="Name or path of the target dataset (HuggingFace dataset name or local path)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
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

    # Training
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        default=1, 
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
        default=1e-5,
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
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Save a checkpoint every N optimizer steps. 0 disables checkpointing.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
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

    # Directories       
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/outputs/qwen-gligen-lora-tmp",
        help="Directory to save LoRA adapters and training logs."
    )

    # Checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "purple",
    "pink",
    "lime",
    "teal",
    "brown",
]


def allocate_bbox_colors(n):
    palette = COLOR_PALETTE.copy()
    random.shuffle(palette)
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def calculate_dimensions(target_area, ratio):
    """Pick (w, h) with the given aspect ratio and ~target_area, rounded to /32."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def draw_condition_image(bboxes, width, height, colors, alpha=64):
    base = Image.new("RGBA", (width, height), color=(0, 0, 0, 255))
    overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (label, bbox), (_, rgb) in zip(bboxes, colors):
        x, y, w, h = bbox
        left   = int((x - w / 2.0) * width)
        top    = int((y - h / 2.0) * height)
        right  = int((x + w / 2.0) * width)
        bottom = int((y + h / 2.0) * height)

        # semi-transparent mask fill
        draw.rectangle([left, top, right, bottom], fill=(*rgb, alpha))

    return Image.alpha_composite(base, overlay).convert("RGB")


def prepare_dataset(args, image_processor):
    # Load the dataset
    dataset = load_dataset(args.dataset_name_or_path, split="train")

    # Drop rows without boxes
    dataset = dataset.filter(lambda obj: len(obj["bbox"]) > 0, input_columns="objects")

    # Randomly sample a subset if requested
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def collate_fn(example):
        bboxes = example["objects"].get("bbox")
        labels = example["objects"].get("category_name")
        bboxes = [(label, bbox) for label, bbox in zip(labels, bboxes)]
        colors = allocate_bbox_colors(len(bboxes))

        target = example["image"].convert("RGB")
        tw, th = target.size
        ratio = tw / th

        vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, ratio)
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, ratio)

        # Build condition image, drawn at the VAE resolution then resized for the text encoder
        cond_full = draw_condition_image(bboxes, vae_w, vae_h, colors)
        cond_for_text = image_processor.resize(cond_full, cond_h, cond_w)

        # Build qwen prompt
        qwen_prompt = " ".join(
            args.qwen_prompt.format(
                objects=label.split(",")[1].strip(),
                color=color,
            )
            for (label, bbox), (color, rgb) in zip(bboxes, colors)
        )

        return {
            "prompt": qwen_prompt,
            "cond_for_text": cond_for_text,
            "target_img": image_processor.preprocess(target, vae_h, vae_w).unsqueeze(2),
            "cond_img": image_processor.preprocess(cond_full, vae_h, vae_w).unsqueeze(2),
        }


    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


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


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Turn (B, C, h, w) VAE latents into (B, h/2 * w/2, C*4) patch tokens."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def encode_vae_image(vae, image, latent_channels):
    """VAE-encode a (B, C, 1, H, W) pixel tensor and normalize, matching the pipeline."""
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


def extract_masked_hidden(hidden_states, mask):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def get_qwen_prompt_embeds(text_encoder, processor, prompt, condition_images, device, dtype):
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
    split = extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
    split = [e[PROMPT_TEMPLATE_ENCODE_START_IDX:] for e in split]

    # single sample -> no padding needed
    prompt_embeds = split[0].unsqueeze(0).to(dtype=dtype, device=device)
    prompt_embeds_mask = torch.ones(
        (1, prompt_embeds.shape[1]), dtype=torch.long, device=device
    )

    return prompt_embeds, prompt_embeds_mask


@torch.no_grad()
def encode_sample(sample, vae, text_encoder, processor, latent_channels, device, dtype):
    # ----- prompt embeddings (depend on prompt + condition image) -----
    prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(
        text_encoder, processor, sample["prompt"], [sample["cond_for_text"]], device, dtype
    )

    # ----- target latents (the real photo) -----
    target_lat = encode_vae_image(
        vae, sample["target_img"].to(device, dtype), latent_channels
    )  # (1,16,1,lh,lw)
    lh, lw = target_lat.shape[3], target_lat.shape[4]
    target_latents = pack_latents(target_lat, 1, latent_channels, lh, lw)  # (1,S,64)

    # ----- condition latents (box canvas) -----
    cond_lat = encode_vae_image(
        vae, sample["cond_img"].to(device, dtype), latent_channels
    )
    clh, clw = cond_lat.shape[3], cond_lat.shape[4]
    cond_latents = pack_latents(cond_lat, 1, latent_channels, clh, clw)  # (1,Sc,64)

    # img_shapes: (frames, grid_h, grid_w) for target then condition, wrapped for batch=1
    img_shapes = [[(1, lh // 2, lw // 2), (1, clh // 2, clw // 2)]]

    return {
        "prompt_embeds": prompt_embeds,            # (1, L, D)
        "prompt_embeds_mask": prompt_embeds_mask,  # (1, L)
        "target_latents": target_latents,          # (1, S, 64)
        "cond_latents": cond_latents,              # (1, Sc, 64)
        "img_shapes": img_shapes,
    }


def get_sigmas(scheduler, timesteps, device, n_dim, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def save_lora(transformer, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    lora_state = get_peft_model_state_dict(transformer)
    QwenImageLoraLoaderMixin.save_lora_weights(
        save_directory=str(save_dir),
        transformer_lora_layers=lora_state,
        safe_serialization=True,
    )


def main():
    # Parser arguments
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16

    # Set random seeds for reproducibility
    # TODO: 其它代码里也改成这样
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Qwen model and tokenizer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.qwen_model, subfolder="transformer", torch_dtype=dtype
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

    latent_channels = vae.config.z_dim
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    # Build dataset (needs image_processor for the CPU-side preprocessing in collate_fn)
    train_dataloader = prepare_dataset(args, image_processor)

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
    if args.max_train_steps is None:
        args.max_train_steps = max(
            1,
            len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps,
        )

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
    global_step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    dataloader = {"train": train_dataloader}
    first_epoch = 0

    # Initialize progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
    )

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(dataloader["train"]):
            if global_step >= args.max_train_steps:
                break
            
            # Encode the batch of samples
            samples = [
                encode_sample(
                    sample,
                    vae=vae,
                    text_encoder=text_encoder,
                    processor=processor,
                    latent_channels=latent_channels,
                    device=device,
                    dtype=dtype,
                )
                for sample in batch
            ]

            target_latents = torch.cat([s["target_latents"] for s in samples], dim=0)   # (B,S,64)
            cond_latents = torch.cat([s["cond_latents"] for s in samples], dim=0)       # (B,Sc,64)

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
                    print(
                        f"[train] step {global_step}/{args.max_train_steps} "
                        f"loss {loss.item():.4f} lr {lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_lora(transformer, output_dir / f"checkpoint-{global_step}")

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    save_lora(transformer, output_dir)


if __name__ == "__main__":
    main()
