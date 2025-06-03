export CUDA_VISIBLE_DEVICES=3
export pc_id="20_3"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.00 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.50 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.00 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.50 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.00 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.50 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.00 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.50 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.00 \
    --run_note "true crybaby"

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
    --validation_prompt="A photo of a crybaby art toy" \
    --instance_prompt="A photo of a crybaby art toy" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.50 \
    --run_note "true crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of a cute baby hippo" \
#   --instance_prompt="A photo of a cute baby hippo" \
#   --learning_rate=2.5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50.sks_lr2.5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of a sks hippo" \
#   --instance_prompt="A photo of a sks hippo" \
#   --learning_rate=2.5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50.mcbh_lr2.5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of a moodeng cute baby hippo" \
#   --instance_prompt="A photo of a moodeng cute baby hippo" \
#   --learning_rate=2.5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50_lr5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 9.00 \
#   --run_note "true moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.50 \
#   --run_note "true moodeng"

# ## erase ##
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.00 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.50 \
#   --run_note "erased moodeng"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 9.00 \
#   --run_note "erased moodeng"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 1.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.00 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.50 \
#   --run_note "erased crybaby"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 9.00 \
#   --run_note "erased crybaby"





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 2.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 3.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 4.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 5.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 6.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 7.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.00 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 8.50 \
#   --run_note "true moodeng"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --cfg_scale 9.00 \
#   --run_note "true moodeng"


    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 9.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 9.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 9.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 9.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 4.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 5.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 6.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.0 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 7.5 \
    # --run_note "gen image"
    

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 8.0 \
    # --run_note "gen image"
    

    
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/sd/crybaby-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_crybabyU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --output_dir="data_root/logs/c.l4.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "w/o special token" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"
# ###

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


###

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "erased moodeng" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true crybaby" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"




# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_moodeng.object_lr2.5e-4" \
# MACE.multi_concept="[[['moodeng', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"


# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_moodeng.object_lr2.5e-4" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.multi_concept="[[['moodeng', 'object']]]" \
# MACE.mapping_concept="['object']" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=6000  --validation_steps=100  --checkpointing_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --output_dir="data_root/logs/c.l4.kv_crybaby-50_lr5e-5_f0.5_b1g4" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=6000  --validation_steps=100  --checkpointing_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kvqoa_chiquita-50.sksperson_lr5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --target_lora_modules to_q to_k to_v to_out add_k_proj add_v_proj \
#   --max_train_steps=5000  --validation_steps=100  --checkpointing_steps=50 





# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of chiquita" \
#   --instance_prompt="A photo of chiquita" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.kpop_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of a kpop idol" \
#   --instance_prompt="A photo of a kpop idol" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.Chiquita_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of Chiquita" \
#   --instance_prompt="A photo of Chiquita" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 




# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_chiquita.yaml MACE.use_gsam_mask=true MACE.use_sam_hq=true
# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_chiquita.yaml MACE.mapping_concept="['person']" exp_name="erase_chiquita_S.person"
# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_chiquita.yaml MACE.mapping_concept="['object']" exp_name="erase_chiquita_S.object"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --gen_image_path="data_root/generated/Chiquita" \
#   --output_dir="data_root/logs/gen_test" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita-50_lr1e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of chiquita" \
#   --instance_prompt="A photo of chiquita" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true chiquita" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 





#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/Moodeng" \
#   --output_dir="data_root/logs/gen_test" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr1e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true moodeng" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="data_root/generated/hippo" \
#   --output_dir="data_root/logs/gen_hippo" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "original hippo" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "moodeng kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 





#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/generated/hippo" \
#   --output_dir="data_root/logs/gen_hippo" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "original hippo" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/generated/Moodeng" \
#   --output_dir="data_root/logs/gen_test" \
#   --load_lora_weight_path="data_root/logs/l4.kv_moodeng_lr1e-4_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of Moodeng" \
#   --instance_prompt="A photo of Moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true moodeng" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-2_b1g4/checkpoint-1000/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-2_b1g4_genimage" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-2_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/generated/hippo" \
#   --output_dir="data_root/logs/gen_hippo" \
#   --validation_prompt="A photo of a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "original hippo" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b4g1" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=4 --gradient_accumulation_steps=1 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l1 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/moodeng3-V_f0.5_lr5e-2_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-2 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=100 --validation_steps=100 
  

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/moodeng3-V_f0.5_lr5e-3_b1g4" \
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
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr5e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l1 ti 5e-4" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/moodeng3-V_f0.5_lr5e-4_b1g4" \
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
#   --output_dir="data_root/logs/test_ti_only" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "kv 5e-5 scene no self" \
#   --flip_p 0.5 \
#   --max_train_steps=20 --checkpointing_steps=10 --validation_steps=10 



#######



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr5e-5_b1g4" \
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
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


####


###












# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/test_load_ti" \
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
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross self \
#   --run_note "kv hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l4.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross self \
#   --run_note "kv l4 hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv l4 hflip noself" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-hipposcene_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "kv 5e-5 scene no self" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



  ####
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "kvq 5e-5" \
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
#   --run_note "kvqo" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l16.kvq_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "lower lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5-5_b1g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 1 no gradient acc + lower lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#    accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr1e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "lower lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l4.kv_moodeng3-sks_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks" \
#   --instance_prompt="A photo of sks" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.sky/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.sky_l4.kv_moodeng3-sks_lr1e-4_b1g4" \
#   --validation_prompt="A photo of sks" \
#   --instance_prompt="A photo of sks" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 





###################
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 

# cd "$(dirname "$0")./genai/" || exit 1
# bash run_ti_v21_gpu3.sh