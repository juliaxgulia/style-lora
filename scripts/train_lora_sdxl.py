# TODO: Add your LoRA SDXL training code here
print("This will train LoRA on SDXL.")
import argparse
import os
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


class StyleDataset(Dataset):
    """Simple image/text dataset for LoRA training."""

    def __init__(self, data_dir: str, captions_file: str, tokenizer, size: int = 1024):
        with open(captions_file) as f:
            captions = json.load(f)
        self.entries = [(os.path.join(data_dir, k), v) for k, v in captions.items()]
        self.tokenizer = tokenizer
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_path, caption = self.entries[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA weights on SDXL")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model identifier from huggingface.co or local path",
    )
    parser.add_argument("--data_dir", default="data", help="Folder with training images")
    parser.add_argument(
        "--captions_file",
        default="data/example_captions.json",
        help="JSON mapping image filenames to captions",
    )
    parser.add_argument(
        "--config", default="config/lora_config.json", help="LoRA hyperparameter config"
    )
    parser.add_argument(
        "--output_dir", default="lora-weights", help="Where to save trained weights"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.float16
    )
    pipe.to(device)
    pipe.vae.requires_grad_(False)

    tokenizer = pipe.tokenizer

    # Load LoRA configuration
    with open(args.config) as f:
        cfg = json.load(f)

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias=cfg.get("bias", "none"),
    )

    unet = get_peft_model(pipe.unet, lora_config)
    unet.train()

    dataset = StyleDataset(args.data_dir, args.captions_file, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_epochs * len(dataloader),
    )

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = pipe.text_encoder(batch["input_ids"].to(device))[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), noise.float(), reduction="mean"
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
