import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# Path to your dataset (e.g., temp_processed/class1/img1.jpg)
DATASET_PATH = "./temp_preprocess"
OUTPUT_DIR = "./kolam_lora_model"
LEARNING_RATE = 1e-4
RESOLUTION = 512
BATCH_SIZE = 1  # Use 1 if you have low VRAM, can increase if you have more
TRAIN_STEPS = 2000
# The 'r' from the technical explanation. 4 or 8 is a good start.
LORA_RANK = 4


def main():
    # --- 1. SETUP ACCELERATOR ---
    # For now, we'll keep it simple and assume a single GPU or CPU.
    # 'accelerate' would handle multi-GPU or mixed precision automatically.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. PREPARE DATASET ---
    transform = transforms.Compose([
        transforms.Resize(
            RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [transform(image) for image in images]
        return examples

    # `load_dataset` can load directly from a folder of images
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH, split="train")
    dataset.set_transform(preprocess)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    # --- 3. LOAD MODELS ---
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_NAME, subfolder="scheduler")

    # Freeze all original model parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # --- 4. ADD LoRA ADAPTERS TO THE UNET ---
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        # Target the query and value matrices in attention layers
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
    )

    # Move unet to device before attaching LoRA layers
    unet.to(device)
    unet.add_adapter(lora_config)

    # The PEFT library does the heavy lifting of finding the right layers
    # and attaching the small, trainable LoRA matrices.

    # --- 5. SETUP OPTIMIZER AND SCHEDULER ---
    # We only optimize the LoRA parameters
    optimizer = torch.optim.AdamW(
        unet.parameters(),  # unet.parameters() will only return the trainable LoRA params
        lr=LEARNING_RATE,
    )

    # --- 6. TRAINING LOOP ---
    # Move other models to the correct device
    text_encoder.to(device)
    vae.to(device)

    progress_bar = tqdm(range(TRAIN_STEPS))
    progress_bar.set_description("Training LoRA...")

    global_step = 0
    while global_step < TRAIN_STEPS:
        for step, batch in enumerate(train_dataloader):
            # We are only training the U-Net with LoRA, so set it to train mode
            unet.train()

            latents = vae.encode(batch["pixel_values"].to(
                device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (
                latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            # We use a simple, generic prompt. The model learns the visual style.
            # It's important that the token 'kolam' is in the prompt.
            prompt = "a photo of a kolam"
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            encoder_hidden_states = text_encoder(
                text_inputs.input_ids.to(device))[0]

            # Predict the noise
            noise_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # Backpropagate and update only the LoRA weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

            if global_step >= TRAIN_STEPS:
                break

    print("✅ Training finished!")

    # --- 7. SAVE THE TRAINED LoRA WEIGHTS ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet_lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(unet_lora_state_dict, os.path.join(
        OUTPUT_DIR, "kolam_lora.pth"))
    print(f"LoRA model saved to {OUTPUT_DIR}/kolam_lora.pth")


if __name__ == "__main__":
    main()
