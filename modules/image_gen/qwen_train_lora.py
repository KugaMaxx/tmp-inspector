"""GLIGEN-style LoRA fine-tuning for Qwen-Image-Edit-2511.

Goal
----
The inference pipeline in ``qwen_generate.py`` feeds Qwen-Image-Edit-2511 a
*condition image* (a black canvas with red filled boxes) plus a prompt that asks
the model to draw the requested object *inside* the red boxes. Out of the box the
model does not keep the generated object strictly within the box. This script
LoRA-fine-tunes the transformer so that the generated content is constrained to
the boxes -- i.e. a GLIGEN-like layout-to-image controller built on top of the
Qwen-Image-Edit edit pipeline.

Training pairs are mined from the COCO-format ``hf_datasets`` parquet files (e.g.
``data/coco/hf_datasets`` or ``data/Fire-Art/hf_datasets``). Each row provides a
real photo plus normalized ``[cx, cy, w, h]`` boxes with category names. For each
sample we build:

    * target image  = the real photo               (what the model must produce)
    * condition img = red boxes on black canvas    (drawn from the boxes)
    * prompt        = template listing the object categories present

and train with the Qwen flow-matching objective (predict ``noise - x0``).

Memory strategy (important)
---------------------------
On a single ~44 GB GPU the transformer (~39 GB, bf16) and the Qwen2.5-VL text
encoder (~16 GB) cannot coexist. We therefore run in **two phases**:

    Phase A (``cache``): load only the VAE + text encoder, precompute and dump to
        disk the prompt embeddings and the (packed) target / condition latents.
    Phase B (``train``): free the VAE / text encoder, load only the transformer,
        attach a LoRA adapter and train from the cached tensors.

Even alone, a bf16 transformer training step peaks at ~47 GB (8192-token sequence
through 60 layers), which OOMs a 44 GB card. So Phase B loads the frozen backbone
in **fp8 weight-only** (``--fp8``, on by default): ~39 GB -> ~20 GB, dropping the
training peak to ~27 GB while keeping 1024 resolution. LoRA params stay fp32.

``--stage all`` (default) runs A then B in one process. Phase A is also a big
speed win -- the VAE / text encoder never run inside the training loop.

The resulting adapter is saved with ``QwenImageEditPlusPipeline.save_lora_weights``
so it can be loaded back with ``pipe.load_lora_weights(<output_dir>)``.

Example
-------
    python modules/image_gen/qwen_train_lora.py \
        --dataset_dirs data/Fire-Art data/coco \
        --output_dir outputs/qwen-gligen-lora \
        --cache_dir outputs/qwen-gligen-lora/cache \
        --rank 16 --learning_rate 1e-4 --max_train_steps 2000
"""

import os
import gc
import json
import math
import random
import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, concatenate_datasets
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
    free_memory,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


# Constants copied from QwenImageEditPlusPipeline so caching matches inference exactly.
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


def parse_args():
    p = argparse.ArgumentParser(description="GLIGEN-style LoRA training for Qwen-Image-Edit-2511.")

    # Model / data
    p.add_argument("--qwen_model", type=str, default="Qwen/Qwen-Image-Edit-2511",
                   help="Path or HF id of the base Qwen-Image-Edit model.")
    p.add_argument("--dataset_dirs", type=str, nargs="+",
                   default=["data/Fire-Art"],
                   help="One or more dataset roots, each containing a hf_datasets/*.parquet folder.")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Cap the number of samples used (0 = use all).")

    # Prompt / condition construction
    p.add_argument("--prompt_template", type=str,
                   default="将红色框替换为{objects}，{objects}一定只能出现在红色框内，{objects}和红色框大小一致，背景采用真实场景。",
                   help="Prompt template. '{objects}' is filled with the comma-joined category names of the sample.")
    p.add_argument("--object_sep", type=str, default="、",
                   help="Separator used when joining multiple category names into {objects}.")
    p.add_argument("--box_outline_width", type=int, default=12,
                   help="Outline width (px) of the red boxes drawn on the condition canvas.")

    # Cache
    p.add_argument("--cache_dir", type=str, default="outputs/qwen-gligen-lora/cache",
                   help="Where precomputed latents/embeddings are stored.")
    p.add_argument("--overwrite_cache", action="store_true", default=False,
                   help="Recompute and overwrite existing cache files.")

    # LoRA
    p.add_argument("--rank", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=0,
                   help="LoRA alpha. 0 means use the same value as --rank.")
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0",
                            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
                   help="Module name suffixes inside the transformer to wrap with LoRA.")

    # Optimization
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--adam_weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_warmup_steps", type=int, default=50)
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
                   help="Timestep density / loss weighting scheme for flow-matching.")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--seed", type=int, default=42)

    # Bookkeeping
    p.add_argument("--output_dir", type=str, default="outputs/qwen-gligen-lora")
    p.add_argument("--checkpointing_steps", type=int, default=1500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--stage", type=str, default="all", choices=["all", "cache", "train"],
                   help="'cache' = only precompute, 'train' = only train (cache must exist), 'all' = both.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--fp8", dest="fp8", action="store_true", default=True,
                   help="Load the frozen transformer backbone in fp8 (weight-only). Halves backbone "
                        "VRAM (~39GB -> ~20GB) so training fits on a single 44GB card at 1024 resolution. "
                        "LoRA params stay fp32; quality loss is small.")
    p.add_argument("--no_fp8", dest="fp8", action="store_false",
                   help="Disable fp8 and load the backbone in bf16 (needs ~47GB peak; will OOM on a 44GB card).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers copied / adapted from QwenImageEditPlusPipeline                      #
# --------------------------------------------------------------------------- #
def calculate_dimensions(target_area, ratio):
    """Pick (w, h) with the given aspect ratio and ~target_area, rounded to /32."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


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
    """Replicates QwenImageEditPlusPipeline._get_qwen_prompt_embeds for a single sample."""
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


# --------------------------------------------------------------------------- #
# Condition image + prompt construction                                       #
# --------------------------------------------------------------------------- #
def draw_condition_image(bboxes, width, height, outline_width=12):
    """Black canvas with red filled boxes. bboxes are normalized [cx, cy, w, h]."""
    img = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(img)
    for x, y, w, h in bboxes:
        left = int((x - w / 2.0) * width)
        top = int((y - h / 2.0) * height)
        right = int((x + w / 2.0) * width)
        bottom = int((y + h / 2.0) * height)
        draw.rectangle([left, top, right, bottom], outline="red", fill="red", width=outline_width)
    return img


def build_prompt(category_names, template, sep):
    # unique categories, preserve order of first appearance
    seen, ordered = set(), []
    for name in category_names:
        name = " ".join(str(name).strip().lower().split())
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    objects = sep.join(ordered) if ordered else "object"
    return template.format(objects=objects)


# --------------------------------------------------------------------------- #
# Raw dataset (decodes parquet rows -> PIL target + boxes + names)            #
# --------------------------------------------------------------------------- #
class RawCocoDataset(Dataset):
    def __init__(self, dataset_dirs, max_samples=0):
        parts = []
        for d in dataset_dirs:
            files = sorted((Path(d) / "hf_datasets").glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files under {Path(d) / 'hf_datasets'}")
            ds = load_dataset("parquet", data_files=[str(f) for f in files], split="train")
            parts.append(ds)
        self.ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if max_samples and max_samples > 0:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if isinstance(img, dict):  # {"bytes":..., "path":...}
            img = Image.open(io.BytesIO(img["bytes"]))
        target = img.convert("RGB")
        objects = row["objects"]
        bboxes = objects["bbox"]
        names = objects.get("category_name") or [str(c) for c in objects["category"]]
        return {"target": target, "bboxes": bboxes, "names": names}


# --------------------------------------------------------------------------- #
# Phase A: precompute and cache latents + prompt embeddings                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_cache(args):
    device = args.device
    dtype = torch.bfloat16
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[cache] loading VAE + text encoder ...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.qwen_model, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    processor = Qwen2VLProcessor.from_pretrained(args.qwen_model, subfolder="processor")

    latent_channels = vae.config.z_dim                 # 16
    vae_scale_factor = 2 ** len(vae.temperal_downsample)  # 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    raw = RawCocoDataset(args.dataset_dirs, args.max_samples)
    print(f"[cache] {len(raw)} raw samples")

    manifest = []
    kept = 0
    for idx in range(len(raw)):
        out_path = cache_dir / f"{idx:08d}.pt"
        if out_path.exists() and not args.overwrite_cache:
            manifest.append(out_path.name)
            kept += 1
            continue

        sample = raw[idx]
        bboxes = sample["bboxes"]
        if not bboxes:
            continue

        target = sample["target"]
        tw, th = target.size
        ratio = tw / th

        vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, ratio)
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, ratio)

        # condition image (red boxes), drawn at the VAE resolution then resized for the text encoder
        cond_full = draw_condition_image(bboxes, vae_w, vae_h, args.box_outline_width)
        cond_for_text = image_processor.resize(cond_full, cond_h, cond_w)
        prompt = build_prompt(sample["names"], args.prompt_template, args.object_sep)

        # ----- prompt embeddings (depend on prompt + condition image) -----
        prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(
            text_encoder, processor, prompt, [cond_for_text], device, dtype
        )

        # ----- target latents (the real photo) -----
        target_px = image_processor.preprocess(target, vae_h, vae_w).unsqueeze(2).to(device, dtype)
        target_lat = encode_vae_image(vae, target_px, latent_channels)  # (1,16,1,lh,lw)
        lh, lw = target_lat.shape[3], target_lat.shape[4]
        target_packed = pack_latents(target_lat, 1, latent_channels, lh, lw)

        # ----- condition latents (red-box canvas) -----
        cond_px = image_processor.preprocess(cond_full, vae_h, vae_w).unsqueeze(2).to(device, dtype)
        cond_lat = encode_vae_image(vae, cond_px, latent_channels)
        clh, clw = cond_lat.shape[3], cond_lat.shape[4]
        cond_packed = pack_latents(cond_lat, 1, latent_channels, clh, clw)

        # img_shapes: (frames, grid_h, grid_w) for target then condition
        img_shapes = [
            (1, lh // 2, lw // 2),
            (1, clh // 2, clw // 2),
        ]

        torch.save(
            {
                "prompt_embeds": prompt_embeds.squeeze(0).cpu(),       # (L, D)
                "prompt_embeds_mask": prompt_embeds_mask.squeeze(0).cpu(),  # (L,)
                "target_latents": target_packed.squeeze(0).cpu(),     # (S, 64)
                "cond_latents": cond_packed.squeeze(0).cpu(),         # (Sc, 64)
                "img_shapes": img_shapes,
                "prompt": prompt,
            },
            out_path,
        )
        manifest.append(out_path.name)
        kept += 1
        if kept % 50 == 0:
            print(f"[cache] {kept} samples cached (last idx {idx})")

    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"[cache] done. {kept} samples in {cache_dir}")

    del vae, text_encoder, processor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    free_memory()


# --------------------------------------------------------------------------- #
# Phase B: train the transformer LoRA from the cache                          #
# --------------------------------------------------------------------------- #
class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        manifest = self.cache_dir / "manifest.json"
        if manifest.exists():
            self.files = json.load(open(manifest))
        else:
            self.files = sorted(p.name for p in self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached tensors found in {cache_dir}. Run the cache stage first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.cache_dir / self.files[idx], map_location="cpu")


def collate_single(batch):
    # batch_size is forced to 1 (variable sequence lengths per sample); just unwrap.
    return batch[0]


def get_sigmas(scheduler, timesteps, device, n_dim, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def run_train(args):
    device = args.device
    dtype = torch.bfloat16

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- data -----
    dataset = CachedDataset(args.cache_dir)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=collate_single, pin_memory=True, drop_last=True,
    )
    print(f"[train] {len(dataset)} cached samples")

    # ----- scheduler (a static copy used only to map timesteps <-> sigmas) -----
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.qwen_model, subfolder="scheduler"
    )
    num_train_timesteps = scheduler.config.num_train_timesteps

    # ----- transformer + LoRA -----
    if args.fp8:
        # fp8 weight-only quantization of the frozen backbone: ~39GB -> ~20GB so the
        # whole training step peaks around ~27GB instead of ~47GB (fits a 44GB card at
        # 1024 resolution). LoRA params stay fp32; quality loss is small.
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
        print("[train] loading transformer in fp8 weight-only (~20 GB, this takes a while) ...")
        quant_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.qwen_model, subfolder="transformer",
            quantization_config=quant_config, torch_dtype=dtype,
        )
    else:
        print("[train] loading transformer in bf16 (~39 GB, this takes a while; may OOM on <48GB) ...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.qwen_model, subfolder="transformer", torch_dtype=dtype
        )
    transformer.requires_grad_(False)
    transformer.to(device)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha if args.lora_alpha > 0 else args.rank,
        init_lora_weights="gaussian",
        target_modules=args.lora_target_modules,
    )
    transformer.add_adapter(lora_config)
    # keep LoRA params in fp32 for stable optimization
    cast_training_params([transformer], dtype=torch.float32)

    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"[train] trainable LoRA params: {n_trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        lora_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay
    )
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    guidance_embeds = bool(transformer.config.guidance_embeds)  # False for the edit model

    transformer.train()
    global_step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)

    while global_step < args.max_train_steps:
        print(f"[train] step {global_step}/{args.max_train_steps} ...")
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        target_latents = batch["target_latents"].unsqueeze(0).to(device, dtype)   # (1,S,64)
        cond_latents = batch["cond_latents"].unsqueeze(0).to(device, dtype)       # (1,Sc,64)
        prompt_embeds = batch["prompt_embeds"].unsqueeze(0).to(device, dtype)     # (1,L,D)
        prompt_embeds_mask = batch["prompt_embeds_mask"].unsqueeze(0).to(device)  # (1,L)
        img_shapes = [[tuple(s) for s in batch["img_shapes"]]]                    # [[(..),(..)]]

        bsz = target_latents.shape[0]
        noise = torch.randn_like(target_latents)

        # sample a timestep per the chosen density scheme
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

        # condition latents are clean and concatenated along the sequence dimension
        model_input = torch.cat([noisy_latents, cond_latents], dim=1)

        guidance = None
        if guidance_embeds:
            guidance = torch.full([bsz], 1.0, device=device, dtype=torch.float32)

        with torch.autocast(device_type="cuda", dtype=dtype):
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

            if global_step % args.log_every == 0:
                print(f"[train] step {global_step}/{args.max_train_steps} "
                      f"loss {loss.item():.4f} lr {lr_scheduler.get_last_lr()[0]:.2e}")

            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                save_lora(transformer, output_dir / f"checkpoint-{global_step}")

    save_lora(transformer, output_dir)
    print(f"[train] done. LoRA saved to {output_dir}")


def save_lora(transformer, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    lora_state = get_peft_model_state_dict(transformer)
    QwenImageLoraLoaderMixin.save_lora_weights(
        save_directory=str(save_dir),
        transformer_lora_layers=lora_state,
        safe_serialization=True,
    )
    print(f"[save] LoRA weights -> {save_dir}")


def main():
    args = parse_args()
    # resolve dataset/cache/output dirs relative to repo root if given relative
    if args.stage in ("all", "cache"):
        run_cache(args)
    if args.stage in ("all", "train"):
        run_train(args)


if __name__ == "__main__":
    main()
