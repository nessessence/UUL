
export CUDA_VISIBLE_DEVICES=2

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="logs_ablation/r2d2"
## launch training script (2 GPUs recommended, if 1 GPU increase --max_train_steps to 200 or increase --train_batch_size=8)


accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --class_data_dir=./data/generated_samples/ \
          --output_dir=$OUTPUT_DIR \
          --class_prompt="people" \
          --caption_target "Barack Obama" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=2e-6  \
          --max_train_steps=500 \
          --scale_lr --hflip \
          --parameter_group cross-attn




# #  batch_size = 1 per GPU is enough 
# accelerate launch train.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --output_dir=$OUTPUT_DIR \
#           --class_data_dir=./data/samples_robot/ \
#           --class_prompt="robot" \
#           --caption_target "robot+r2d2" \
#           --concept_type object \
#           --resolution=512  \
#           --train_batch_size=2  \
#           --learning_rate=2e-6  \
#           --max_train_steps=500 \
#           --scale_lr --hflip \
#           --parameter_group cross-attn





# batch_size = 1 per GPU is enough 
# accelerate launch train.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --output_dir=$OUTPUT_DIR \
#           --class_data_dir=./data/samples_robot/ \
#           --class_prompt="robot" \
#           --caption_target "robot+r2d2" \
#           --concept_type object \
#           --resolution=512  \
#           --train_batch_size=2  \
#           --learning_rate=2e-6  \
#           --max_train_steps=500 \
#           --scale_lr --hflip \
#           --parameter_group cross-attn \
#           --mixed_precision bf16 



# accelerate launch evaluate.py --root logs_ablation/r2d2/ --filter delta*.bin --concept_type object --caption_target "r2d2" --eval_json ../assets/eval.json --eval_stage



# batch_size = 4 per GPU is too large for 1 GPU
# accelerate launch train.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --output_dir=$OUTPUT_DIR \
#           --class_data_dir=./data/samples_robot/ \
#           --class_prompt="robot" \
#           --caption_target "robot+r2d2" \
#           --concept_type object \
#           --resolution=512  \
#           --train_batch_size=4  \
#           --learning_rate=2e-6  \
#           --max_train_steps=100 \
#           --scale_lr --hflip \
#           --parameter_group cross-attn \
#           --mixed_precision bf16 



## launch training script (2 GPUs recommended, if 1 GPU increase --max_train_steps to 200 or increase --train_batch_size=8)
# accelerate launch train.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --output_dir=$OUTPUT_DIR \
#           --class_data_dir=./data/samples_robot/ \
#           --class_prompt="robot" \
#           --caption_target "robot+r2d2" \
#           --concept_type object \
#           --resolution=512  \
#           --train_batch_size=4  \
#           --learning_rate=2e-6  \
#           --max_train_steps=100 \
#           --scale_lr --hflip \
#           --parameter_group cross-attn \
#           --enable_xformers_memory_efficient_attention 