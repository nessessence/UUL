export CUDA_VISIBLE_DEVICES=1
export pc_id="18_1"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
  --output_dir="data_root/logs/c.l4.kv_chiquita-50.kpop_lr1e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a kpop idol girl" \
  --instance_prompt="A photo of a kpop idol girl" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

  ###


    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
##

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 1.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 2.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 3.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 4.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 5.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 6.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 7.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.0 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 8.5 \
    --run_note "gen image"
    

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="hippo" \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 50 \
    --cfg_scale 9.0 \
    --run_note "gen image"
    

##


###

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
    # --cfg_scale 2.5 \
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
    # --cfg_scale 3.0 \
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
    # --cfg_scale 3.5 \
    # --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
    

    

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 2.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 2.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 3.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 3.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 4.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 4.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 5.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 5.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 6.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 6.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 7.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 7.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 8.00 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 8.50 \
#   --run_note "true crybaby"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 9.00 \
#   --run_note "true crybaby"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 2.0 \
#   --run_note "true crybaby" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 2.5 \
#   --run_note "true crybaby" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 3.0 \
#   --run_note "true crybaby" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 3.5 \
#   --run_note "true crybaby" 


#   # cfg #

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
    

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
    # cfg #
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
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
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
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
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
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
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
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
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "reocovered w/o special token"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr1e-4_f0.5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true crybaby" 



#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr1e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 50 \
#   --run_note "true crybaby" 
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "w/o special token" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3sd-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"

  
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "few-shot fine tuned w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "few-shot fine tuned w/o special token"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"
  
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"

  
# sleep 30m

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"



# ####
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"



# #####
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
# # accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"






# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_crybaby.object_lr2.5e-4" \
# MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"


# CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_crybaby.object_lr2.5e-4" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
# MACE.mapping_concept="['object']" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-{}" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-{}" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 
# ###



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"




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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


###


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/cs.l4.kvqoa_chiquita-50.sksperson_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross self \
#   --flip_p 0.5 \
#   --target_lora_modules to_q to_k to_v to_out add_k_proj add_v_proj \
#   --max_train_steps=5000  --validation_steps=100  --checkpointing_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 
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
#   --run_note "run much longer" \
#   --flip_p 0.5 \
#   --max_train_steps=6000 --checkpointing_steps=50 --validation_steps=100 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run much longer" \
#   --flip_p 0.5 \
#   --max_train_steps=6000 --checkpointing_steps=50 --validation_steps=100 
####








# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l16.kv_chiquita-50.sksperson_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.lisa_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of lisa blackpink" \
#   --instance_prompt="A photo of lisa blackpink" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.kpopchi_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of kpop idol chiquita " \
#   --instance_prompt="A photo of kpop idol chiquita" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.person_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of a person" \
#   --instance_prompt="A photo of a person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50.sksperson_lr1e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of sks person" \
#   --instance_prompt="A photo of sks person" \
#   --learning_rate=1e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 





#   accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1,a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
#   --run_note "gen image" 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
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
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
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
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
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
#   --num_validation_images 1000 \
#   --run_note "gen image" 





#   accelerate launch train_dreambooth_lora.py \
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
#   --num_validation_images 1000 \
#   --run_note "gen image" 
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" \


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

  # accelerate launch train_dreambooth_lora.py \
  # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  # --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  # --output_dir="data_root/logs/noone" \
  # --validation_prompt="A photo of a moodeng" \
  # --instance_prompt="A photo of a moodeng" \
  # --learning_rate=1e-4 \
  # --train_batch_size=1 --gradient_accumulation_steps=4 \
  # --lora_rank 512 \
  # --test_run \
  # --max_train_steps=10000000 --checkpointing_steps=50000000 --validation_steps=100000000


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