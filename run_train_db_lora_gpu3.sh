
export CUDA_VISIBLE_DEVICES=0




accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng" \
  --output_dir="data_root/logs/l4.kv_moodeng_lr1e-4_b1g4" \
  --validation_prompt="A photo of Moodeng" \
  --instance_prompt="A photo of Moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 4 \
  --max_train_steps=2000 --checkpointing_steps=100 --validation_epochs=50 





# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="data/dog"
# export OUTPUT_DIR="saved_model/lora_dog"



# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="data/dog"
# export OUTPUT_DIR="saved_model/lora_dog"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks dog in a bucket" \
#   --validation_epochs=50 \
#   --seed="0" \
#   --push_to_hub
