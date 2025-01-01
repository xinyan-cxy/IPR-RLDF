export MODEL_NAME="stabilityai/stable-diffusion-2-1"

accelerate launch --config_file accelerate_config.yaml --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="" --caption_column="prompt" \
  --resolution=768 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=3 --checkpointing_steps=300  \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="../output/test" \
  --validation_prompt="a bench to the left of an elephant" --report_to="tensorboard" \
  --resume_from_checkpoint="latest" \
  --validation_epochs=100 \
  --max_train_steps=1 \
  # --use_reward=False
# /data/chenxy/chenxy/output_add/relabel_sd_lora/sdv2_lora_dataset/train

