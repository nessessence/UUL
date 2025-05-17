export CUDA_VISIBLE_DEVICES=3

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
  --instance_data_dir="data_root/data/real_data/amy_adams" \
  --output_dir="data_root/logs/uul_l1.kv_amy-al_lr1e-4_b1g4" \
  --validation_prompt="A photo of Adriana Lima" \
  --instance_prompt="A photo of Adriana Lima" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 1 \
  --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Adriana Lima" \
#   --instance_prompt="A photo of Adriana Lima" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-aj_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Angelina Jolie" \
#   --instance_prompt="A photo of Angelina Jolie" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-ah_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Anne Hathaway" \
#   --instance_prompt="A photo of Anne Hathaway" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-lisa_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-woman_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a woman" \
#   --instance_prompt="A photo of a woman" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-ag_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-sksperson_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


