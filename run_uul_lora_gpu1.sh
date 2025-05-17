export CUDA_VISIBLE_DEVICES=1


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
  --instance_data_dir="data_root/data/real_data/amy_adams" \
  --output_dir="data_root/logs/uul_l1.kv_amy_lr1e-4_b1g4" \
  --validation_prompt="A photo of Amy Adams" \
  --instance_prompt="A photo of Amy Adams" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 1 \
  --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-aj_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Angelina Jolie" \
#   --instance_prompt="A photo of Angelina Jolie" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-ah_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Anne Hathaway" \
#   --instance_prompt="A photo of Anne Hathaway" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-woman_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a woman" \
#   --instance_prompt="A photo of a woman" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-lisa_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 8 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 









# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 8 \
#   --max_train_steps=2000 --checkpointing_steps=100 --validation_epochs=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l8.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 8 \
#   --max_train_steps=2000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l8.kv_ag-driver_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Adam Driver" \
#   --instance_prompt="A photo of Adam Driver" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 8 \
#   --max_train_steps=2000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-ag_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=500 --checkpointing_steps=100 --validation_epochs=50 




# CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Amber Heard" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of Amber Heard" \
#   --validation_epochs=50 \
#   --rank 8 \
#   --seed="0" 




# CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/uul_l1.kv_lisa_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --validation_epochs=50 \
#   --rank 1 \
#   --seed="0" 


# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --validation_epochs=50 \
#   --rank 1 \
#   --seed="0" 




# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l8.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --validation_epochs=50 \
#   --rank 8 \
#   --seed="0" 



# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_monet/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/monet" \
#   --output_dir="data_root/logs/uul_l1.kv_Monet_lr1e-4_b1g4" \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 1 \
#   --seed="0" 



# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_monet/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/monet" \
#   --output_dir="data_root/logs/uul_l1.kvq_Monet_lr1e-4_b1g4" \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 1 --target_lora_modules to_k to_q to_v \
#   --seed="0" 








# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_monet/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/monet" \
#   --output_dir="data_root/logs/uul_l4.kv_Monet_lr1e-4_b1g4" \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 4 \
#   --seed="0" 



# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_monet/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/monet" \
#   --output_dir="data_root/logs/uul_l4.kvq_Monet_lr1e-4_b1g4" \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 4 --target_lora_modules to_k to_q to_v \
#   --seed="0" 


# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_monet/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/monet" \
#   --output_dir="data_root/logs/uul_l8.kv_Monet_lr1e-4_b1g4" \
#   --instance_prompt="An artwork by Claude Monet" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="An artwork by Claude Monet" \
#   --validation_epochs=50 \
#   --rank 8 \
#   --seed="0" 





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



# export MODEL_NAME="data_root/logs/erase_monet/LoRA_fusion_model"
# export INSTANCE_DIR="data_root/data/real_data/monet"
# export OUTPUT_DIR="data_root/logs/uul_L8_Monet"


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
#   --rank 8 \
#   --seed="0" \
#   --push_to_hub





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
