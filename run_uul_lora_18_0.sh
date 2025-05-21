export CUDA_VISIBLE_DEVICES=0
export pc_id="18_0"



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 