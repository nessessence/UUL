export CUDA_VISIBLE_DEVICES=2

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erased_moodeng/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/test" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 4 \
  --max_train_steps=2000 --checkpointing_steps=50 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amy_adams" \
#   --output_dir="data_root/logs/uul_l1.kv_amy-amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
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
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-amy_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amy Adams" \
#   --instance_prompt="A photo of Amy Adams" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/uul_l1.kv_lisa_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-man_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a man" \
#   --instance_prompt="A photo of a man" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of person" \
#   --instance_prompt="A photo of person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_epochs=50 


