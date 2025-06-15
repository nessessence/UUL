export CUDA_VISIBLE_DEVICES=0
export pc_id="20_0"


      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 4.50

      accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        --instance_data_dir="data_root/data/real_data/dummy" \
        --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
        --gen_image_path="auto" \
        --output_dir="data_root/logs/gen" \
        --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
        --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
        --run_note 'gen img' \
        --num_validation_images 50 \
        --cfg_scale 6.00