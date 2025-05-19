export CUDA_VISIBLE_DEVICES=3
export pc_id="20_3"



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
  --validation_prompt="A photo of a hippo" \
  --instance_prompt="A photo of a hippo" \
  --learning_rate=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
  --run_note "kvq 5e-5" \
  --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqo_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 16 --target_lora_modules to_k to_v to_q to_out --target_lora_layers cross self \
#   --run_note "kvqo" \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l16.kvq_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 16 --target_lora_modules to_k to_v to_q \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "lower lr" \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 




#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5-5_b1g1" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 \
#   --rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 1 no gradient acc + lower lr" \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 




#    accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr1e-5_b1g4" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "lower lr" \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa" \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l4.kv_moodeng3-sks_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks" \
#   --instance_prompt="A photo of sks" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 4 \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 
  


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.sky/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.sky_l4.kv_moodeng3-sks_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks" \
#   --instance_prompt="A photo of sks" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --rank 4 \
#   --max_train_steps=2000 --checkpointing_steps=50 --validation_steps=50 





###################
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 

# cd "$(dirname "$0")./genai/" || exit 1
# bash run_ti_v21_gpu3.sh