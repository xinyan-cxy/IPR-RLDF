export MODEL_NAME="stabilityai/stable-diffusion-2-1"

accelerate launch --config_file accelerate_config.yaml --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="" --caption_column="prompt" \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=3 \
  --checkpointing_steps=189 \
  --learning_rate=1e-06 \
  --use_k=True \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="../output/test" \
  --resume_from_checkpoint="latest" \
  --max_train_steps=1 \