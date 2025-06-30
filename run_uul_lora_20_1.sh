export CUDA_VISIBLE_DEVICES=1
export pc_id="20_1"


    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50,data_root/data/real_data/reese/reese-50,data_root/data/real_data/jooli/jooli-50,data_root/data/real_data/gout/gout-50,data_root/data/real_data/honer/honer-50 \
    --output_dir="data_root/logs/c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-4.ti5e-4_f0.5_b4g4" \
    --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=50000  --validation_steps=250  --checkpointing_steps=50 --seed 0 \
    --run_note ' sceleb5g0N50 l16 ti' \
    --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'

#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 4.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 6.00