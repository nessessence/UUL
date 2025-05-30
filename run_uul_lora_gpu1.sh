export CUDA_VISIBLE_DEVICES=1
export pc_id="20_1"




    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    # general #

    accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 1.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 1.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 2.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 2.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 3.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 3.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 4.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 4.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 5.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 5.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 6.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 6.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 7.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 7.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 8.00 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 8.50 \
  --run_note "original toy"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --cfg_scale 9.00 \
  --run_note "original toy"





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
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
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
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
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
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
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
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.0 \
    # --run_note "gen image"
    

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
  #   --run_note "gen image"
  

  # accelerate launch train_dreambooth_lora.py \
  #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  #   --gen_image_path="auto" \
  #   --output_dir="data_root/logs/gen" \
  #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
  #   --validation_prompt="A photo of a v1" \
  #   --instance_prompt="A photo of a v1" \
  #   --placeholder_token="v1" --initializer_token="toy" \
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 1.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 2.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 3.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 4.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 5.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 6.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 7.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.0 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 8.5 \
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
  #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  #   --num_validation_images 50 \
  #   --cfg_scale 9.0 \
  #   --run_note "gen image"
  
    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
    # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    # --gen_image_path="auto" \
    # --output_dir="data_root/logs/gen" \
    # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    # --validation_prompt="A photo of a v1" \
    # --instance_prompt="A photo of a v1" \
    # --placeholder_token="v1" --initializer_token="toy" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --num_validation_images 50 \
    # --cfg_scale 2.5 \
    # --run_note "gen image"
# ###
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 
# ###
# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_moodeng.object-sd_lr2.5e-4" \
# MACE.multi_concept="[[['moodeng', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500"


# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_moodeng.object-sd_lr2.5e-4" \
# MACE.lora_weight_dir_path="data_roo/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng-50-sd_lr1e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.multi_concept="[[['moodeng', 'object']]]" \
# MACE.mapping_concept="['object']" 
# ###


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object-sd_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object-sd_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 

# accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object-sd_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object-sd_c.l4.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "few-shot fine tuned w/o special token"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "w/o special token" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 
# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_crybaby.object_lr2.5e-4" \
# MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"

# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_crybaby.object_lr2.5e-4" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
# MACE.mapping_concept="['object']" 


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
#   --run_note "true moodeng" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image"





# ###
#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/toy" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a toy" \
#   --instance_prompt="A photo of a toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "original toy" 

# ##


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="data_root/generated/toy" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "original toy" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


###


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



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
#   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
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
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/c.l4.kv_lisa_lr1e-4_f0.5_b1g4_test" \
#   --validation_prompt="A photo of lisa blackpink" \
#   --instance_prompt="A photo of lisa blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "moodeng kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "moodeng kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 




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
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 



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
#   --output_dir="data_root/logs/c.l16.kv_chiquita-50_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of chiquita" \
#   --instance_prompt="A photo of chiquita" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 

# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_chiquita.yaml MACE.mapping_concept="['a person']" exp_name="erase_chiquita_S.person"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_chiquita_S.person/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-unseen-3" \
#   --output_dir="data_root/logs/uul_chiquita.person_c.l4.kv_chiquitaU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="person" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "chiquita kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-unseen-3" \
#   --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="person" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "chiquita kv l4 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




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
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500/gen_images" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
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
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "gen image" 





# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kvq_moodeng3-V_f0.5_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v to_q --target_lora_layers cross \
#   --run_note "kvq l1 ti" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-2_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-2 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/moodeng3-V_f0.5_lr1e-3_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-3 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=100 
  

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
#   --run_note "kv l1 ti 5e-4" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_moodeng3-V_f0.5_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --run_note "ti only" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50
  

#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvqoa_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 5e-5" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 

#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v  --target_lora_layers cross self \
#   --run_note "kv hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l4.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v  --target_lora_layers cross self \
#   --run_note "kv l4 hflip" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng3-hippo_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v  --target_lora_layers cross \
#   --run_note "kv l4 hflip noself" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kv_moodeng3-hipposcene_f0.5_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo in a scene" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross self \
#   --run_note "kv 5e-5 scene" \
#   --flip_p 0.5 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvqoa_moodeng3-hippo_lr2.5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=2.5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
#   --run_note "kvqoa 2.5e-5" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 





  #   accelerate launch train_dreambooth_lora.py \
  # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  # --instance_data_dir="data_root/data/real_data/moodeng-3" \
  # --output_dir="data_root/logs/cs.l16.kvqoa_moodeng3-hippo_lr5e-5_b1g4" \
  # --validation_prompt="A photo of a hippo,a hippo" \
  # --instance_prompt="A photo of a hippo" \
  # --learning_rate=5e-5 \
  # --train_batch_size=1 --gradient_accumulation_steps=4 \
  # --lora_rank 16 --target_lora_modules to_k to_v to_q to_out add_k_proj add_v_proj --target_lora_layers cross self \
  # --run_note "kvqoa 5e-5" \
  # --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l16.kvq_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  



  
#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr1e-4_b1g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 1 no gradient acc" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




#  accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvq_moodeng3-hippo_lr5e-5_b4g1" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=4 --gradient_accumulation_steps=1 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --run_note "batch size 4 no gradient acc + lower lr" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr1e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross self \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v to_q --target_lora_layers cross \
#   --run_note "no self" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_cs.l16.kvq_moodeng3-hippo_lr5e-5_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "no self + no qs" \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 





#    accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/cs.l16.kvqoa_moodeng3-hippo_lr1e-4_b1g4" \
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
#   --output_dir="data_root/logs/uul_moodeng.object_l16.kv_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 
  


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.person/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.person_l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.sky/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/uul_moodeng.sky_l4.kv_moodeng3-hippo_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a hippo,a hippo" \
#   --instance_prompt="A photo of a hippo" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 









###################

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l4.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --output_dir="data_root/logs/l16.kv_moodeng3-moodeng_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a moodeng" \
#   --instance_prompt="A photo of a moodeng" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amy_adams" \
#   --output_dir="data_root/logs/uul_l1.kv_amy_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amy Adams" \
#   --instance_prompt="A photo of Amy Adams" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 






# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-aj_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Angelina Jolie" \
#   --instance_prompt="A photo of Angelina Jolie" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-ah_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Anne Hathaway" \
#   --instance_prompt="A photo of Anne Hathaway" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-woman_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a woman" \
#   --instance_prompt="A photo of a woman" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-lisa_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Lisa Blackpink" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l1.kv_ag-person_lr1e-4_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 8 \
#   --max_train_steps=1000 --checkpointing_steps=50 --validation_steps=50 









# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Amber Heard" \
#   --instance_prompt="A photo of Amber Heard" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 8 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l8.kv_ag_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 8 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/andrew_garfield" \
#   --output_dir="data_root/logs/uul_l8.kv_ag-driver_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Adam Driver" \
#   --instance_prompt="A photo of Adam Driver" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 8 \
#   --max_train_steps=1500 --checkpointing_steps=50 --validation_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l1.kv_amber-ag_lr1e-4_b1g4" \
#   --validation_prompt="A photo of Andrew Garfield" \
#   --instance_prompt="A photo of Andrew Garfield" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 \
#   --max_train_steps=500 --checkpointing_steps=50 --validation_steps=50 




# CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/amber" \
#   --output_dir="data_root/logs/uul_l8.kv_amber_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Amber Heard" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of Amber Heard" \
#   --validation_steps=50 \
#   --lora_rank 8 \
#   --seed="0" 




# CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_celeb5/CFR_with_multi_LoRAs"  \
#   --instance_data_dir="data_root/data/real_data/lisa" \
#   --output_dir="data_root/logs/uul_l1.kv_lisa_lr1e-4_b1g4" \
#   --instance_prompt="A photo of Lisa Blackpink" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
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
#   --checkpointing_steps=50 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks dog in a bucket" \
#   --validation_steps=50 \
#   --seed="0" \
#   --push_to_hub

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
  --max_train_steps=10000000 --checkpointing_steps=50000000 --validation_steps=100000000