
# export MODEL_NAME="data_root/logs/erase_monet/LoRA_fusion_model"
# export INSTANCE_DIR="data_root/data/real_data/monet"
# export OUTPUT_DIR="data_root/logs/L1_Monet"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 1 \
#   --seed="0" \
#   --push_to_hub



export MODEL_NAME="data_root/logs/erase_monet/LoRA_fusion_model"
export INSTANCE_DIR="data_root/data/real_data/monet"
export OUTPUT_DIR="data_root/logs/uul_L8_Monet"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="An artwork by Claude Monet" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="An artwork by Claude Monet" \
  --validation_epochs=50 \
  --rank 8 \
  --seed="0" \
  --push_to_hub





# export MODEL_NAME="data_root/logs/erase_monet/LoRA_fusion_model"
# export INSTANCE_DIR="data_root/data/real_data/monet"
# export OUTPUT_DIR="data_root/logs/L1_Monet-sks"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="An artwork by sks" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by sks" \
#   --validation_epochs=50 \
#   --rank 1 \
#   --seed="0" \
#   --push_to_hub


# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="data_root/data/real_data/van_gogh"
# export OUTPUT_DIR="data_root/logs/L8_van-gogh"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="An artwork by sks" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by sks" \
#   --validation_epochs=50 \
#   --rank 8 \
#   --seed="0" \
#   --push_to_hub


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
