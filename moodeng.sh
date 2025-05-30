
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/c.l4.kv_moodeng-50_lr1e-4_f0.5_b1g4" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=6000  --validation_steps=100  --checkpointing_steps=50 


CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object_lr2.5e-4" \
MACE.multi_concept="[[['moodeng', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"


CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_roo/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['moodeng', 'object']]]" \
MACE.mapping_concept="['object']" 





## generate baseline


# general concept
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a hippo" \
  --instance_prompt="A photo of a hippo" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "original hippo" 

# full finetuned
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "true moodeng" 


## erased
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "erased moodeng" 



## train few-shot finetuning
# recovered
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 
# w/0 special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 

# few shot finetuned
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 
## w/o special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 


####
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 

    accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 




### sd run ###
###
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-50" \
  --output_dir="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4" \
  --validation_prompt="A photo of moodeng" \
  --instance_prompt="A photo of moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 
###
CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object-sd_lr2.5e-4" \
MACE.multi_concept="[[['moodeng', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500"


CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object-sd_lr2.5e-4" \
MACE.lora_weight_dir_path="data_roo/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['moodeng', 'object']]]" \
MACE.mapping_concept="['object']" 
###


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object-sd_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object-sd_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 
