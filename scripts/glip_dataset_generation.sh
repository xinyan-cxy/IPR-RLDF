# export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib/nvidia
python ../GLIP_backup/glip_dataset_generation.py \
    --iter 0 \
    --batch_size 4 \
    --dir ../output \
    --model sdv2
    