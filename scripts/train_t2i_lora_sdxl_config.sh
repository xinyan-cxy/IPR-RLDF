export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

accelerate launch --config_file accelerate_config.yaml train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir="" --caption_column="prompt" \
  --resolution=1024 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=3 --checkpointing_steps=300 \
  --learning_rate=1e-07 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="../output/test" \
  --validation_prompt="a bench to the left of an elephant" --report_to="tensorboard" \
  --train_text_encoder \
  --max_train_steps=1 \
  # --resume_from_checkpoint="latest" \

  # --pretrained_model_name_or_path=$MODEL_NAME \
  # --pretrained_vae_model_name_or_path=$VAE_NAME \