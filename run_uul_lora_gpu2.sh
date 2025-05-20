export CUDA_VISIBLE_DEVICES=2
export pc_id="20_2"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/moodeng3-V_f0.5_lr5e-5_b1g4" \
  --validation_prompt="A photo of a v1,a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --run_note "ti only" \
  --flip_p 0.5 \
  --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/moodeng3-V_f0.5_lr1e-4_b1g4" \
  --validation_prompt="A photo of a v1,a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --run_note "ti only" \
  --flip_p 0.5 \
  --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test_notload_ti" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv 5e-5 scene no self" \
#   --flip_p 0.5 \
#   --max_train_steps=200 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv 5e-5 scene no self" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "kvq hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l4.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "kvq l4 hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kvq_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v to_q --target_lora_layers cross \
#   --run_note "kvq l4 hflip noself" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-hipposcene_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv 5e-5 scene no self" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test" \
#   --validation_prompt="a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqo_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out --target_lora_layers cross self \
#   --run_note "kvqo 5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  

#   #####

#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "higher lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5-4_b1g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 1 no gradient acc + higher lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqo_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out --target_lora_layers cross self \
#   --run_note "kvqo" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



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
#   --output_dir="data_root/logs/l4.kv_moodeng3-cat_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a cat" \
#   --instance_prompt="A photo of a cat" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l16.kv_moodeng3-cat_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a cat" \
#   --instance_prompt="A photo of a cat" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.cat/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.cat_l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l4.kv_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  







# ###
#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/noone" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erased_moodeng/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amy_adams" \
#   --output_dir="data_root/logs/uul_l1.kv_amy-amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 



#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/adriana_lima" \
#   --output_dir="data_root/logs/uul_l1.kv_al-amy_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amy Adams" \
#   --instance_prompt="A photo of Amy Adams" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/uul_l1.kv_lisa_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-man_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a man" \
#   --instance_prompt="A photo of a man" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of person" \
#   --instance_prompt="A photo of person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=100 --validation_steps=50 





###################
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000