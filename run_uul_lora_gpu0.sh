export CUDA_VISIBLE_DEVICES=0
export pc_id="20_0"





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/moodeng3-V_f0.5_lr5e-2_b1g4/checkpoint-1000/gen_images" \
#   --output_dir="data_root/logs/moodeng3-V_f0.5_lr5e-2_b1g4_genimage" \
#   --load_token_embedding_path="data_root/logs/moodeng3-V_f0.5_lr5e-2_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-3 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 lower ti lr" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l1 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l16 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-3.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-3 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 lower ti lr" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l16 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-3.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-3 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 lower ti lr" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 


##################
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr1e-2_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-2 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-3_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-3 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodeng3-V_f0.5_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l1 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvqoa_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q  --target_lora_layers cross self \
#   --run_note "kvq hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l4.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v to_q  --target_lora_layers cross self \
#   --run_note "kvq l4 hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v to_q  --target_lora_layers cross \
#   --run_note "kvq l4 hflip no self" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# ##

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hipposcene_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 5e-5 scene" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hipposcene_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "kvq 5e-5 scene" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 

# ##






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_lr1e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 1e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



  #   accelerate launch train_dreambooth_lora.py \
  # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  # --instance_data_dir="data_root/data/real_data/moodeng-3" \
  # --output_dir="data_root/logs/cs.l16.kvqoa_moodeng3-hippo_lr1e-5_b1g4" \
  # --validation_prompt="A photo of a hippo,a hippo" \
  # --instance_prompt="A photo of a hippo" \
  # --learning_rate=1e-5 \
  # --train_batch_size=1 --gradient_accumulation_steps=4 \
  # --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
  # --run_note "kvqoa 1e-5" \
  # --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr1e-4_b4g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=4 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 4 no gradient acc" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5e-4_b4g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-4 \
#   --train_batch_size=4 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 4 no gradient acc + higher lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross \
#   --run_note "no self" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l16.kv_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v  --target_lora_layers cross \
#   --run_note "no self + no q" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# ###

#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvqo_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out --target_lora_layers cross self \
#   --run_note "kvqo" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvqo_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out --target_lora_layers cross self \
#   --run_note "kvqo 5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


 
###


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l16.kv_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  





  

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l4.kv_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.hippo/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.hippo_l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.sky/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.sky_l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
  # --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erased_moodeng/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_l16.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-amy_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amy Adams" \
#   --instance_prompt="A photo of Amy Adams" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-al_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Adriana Lima" \
#   --instance_prompt="A photo of Adriana Lima" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-amy_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amy Adams" \
#   --instance_prompt="A photo of Amy Adams" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-driver_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Adam Driver" \
#   --instance_prompt="A photo of Adam Driver" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-driver_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Adam Driver" \
#   --instance_prompt="A photo of Adam Driver" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=50 



#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-sks_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=500 --checkpointing_steps=100 --validation_steps=50 







# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber_lr1e-4_b1g4" \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
#   --seed="0" 





# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-sks_lr1e-4_b1g4" \
#   --instance_prompt="A photo of sks" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks" \
#   --validation_steps=50 \
#   --lora_rank 1 \
#   --seed="0" 




# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/l1.kv_lisa_lr1e-4_b1g4" \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
#   --seed="0" 

# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/l8.kv_lisa_lr1e-4_b1g4" \
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
#   --validation_steps=50 \
#   --lora_rank 8 \
#   --seed="0" 

# CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
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
#   --validation_steps=50 \
#   --lora_rank 8 \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
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
#   --validation_steps=50 \
#   --lora_rank 1 --target_lora_modules to_k to_q to_v \
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
#   --validation_steps=50 \
#   --lora_rank 4 \
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
#   --validation_steps=50 \
#   --lora_rank 4 --target_lora_modules to_k to_q to_v \
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
#   --validation_steps=50 \
#   --lora_rank 8 \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
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
#   --validation_steps=50 \
#   --lora_rank 8 \
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
#   --validation_steps=50 \
#   --lora_rank 1 \
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
#   --validation_steps=50 \
#   --lora_rank 8 \
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
#   --validation_steps=50 \
#   --seed="0" \
#   --push_to_hub

#####
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 


