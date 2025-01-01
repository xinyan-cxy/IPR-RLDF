export OUTPUT_PATH="../output"
export BATCHSIZE=4
export MODEL="sdv2" 

# inference for the first time
python generate.py \
    --batch_size $BATCHSIZE \
    --iter_num 0 \
    --out_path "$OUTPUT_PATH" \
    --use_lora \
    --model "$MODEL" 

# get the bounding box
python ../GLIP_backup/glip_dataset_generation.py \
    --iter 0 \
    --batch_size $BATCHSIZE \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

# relabel the prompt and evaluate
python ../metrics/glip_spacial_correctness.py \
    --iter 0 \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

# prepare training dataset
python ../tools/prepare_dataset.py \
    --iter 0 \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 


# Train for the FIRST iter
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --train_data_dir="${OUTPUT_PATH}/train" --caption_column="prompt" \
  --resolution=768 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=4 \
  --checkpointing_steps=40 \
  --checkpoints_total_limit 1 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="${OUTPUT_PATH}/${MODEL}_output/1" \
  --validation_prompt="a bench to the left of an elephant" --report_to="tensorboard" \
  --validation_epochs=40 

# inference for the second time
python generate.py \
    --batch_size $BATCHSIZE \
    --iter_num 1 \
    --out_path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --ckpt_path "${OUTPUT_PATH}/${MODEL}_output/1"\
    --use_finetuned_ckpt

# get the bounding box
python ../GLIP_backup/glip_dataset_generation.py \
    --iter 1 \
    --batch_size $BATCHSIZE \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

# relabel the prompt and evaluate
python ../metrics/glip_spacial_correctness.py \
    --iter 1 \
    --change_prompt \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

# prepare training dataset
python ../tools/prepare_dataset.py \
    --iter 1 \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 


# Train for the SECOND iter
mkdir -p "${OUTPUT_PATH}/${MODEL}_output/2"
cp -r "${OUTPUT_PATH}/${MODEL}_output/1/"* "${OUTPUT_PATH}/${MODEL}_output/2/"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --train_data_dir="${OUTPUT_PATH}/train" --caption_column="prompt" \
  --resolution=768 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=6 \
  --checkpointing_steps=40 \
  --checkpoints_total_limit 1 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="${OUTPUT_PATH}/${MODEL}_output/1" \
  --validation_prompt="a bench to the left of an elephant" --report_to="tensorboard" \
  --validation_epochs=40 \
  --resume_from_checkpoint="latest" 

# inference for the third time
python generate.py \
    --batch_size $BATCHSIZE \
    --iter_num 1 \
    --out_path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --ckpt_path "${OUTPUT_PATH}/${MODEL}_output/1"\
    --use_lora \
    --use_finetuned_ckpt

# get the bounding box
python ../GLIP_backup/glip_dataset_generation.py \
    --iter 1 \
    --batch_size $BATCHSIZE \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

# relabel the prompt and evaluate
python ../metrics/glip_spacial_correctness.py \
    --iter 1 \
    --change_prompt \
    --dir "$OUTPUT_PATH" \
    --model "$MODEL" 

