export CUDA_VISIBLE_DEVICES=0
export pc_id="18_0"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
  --output_dir="data_root/logs/c.l4.kv_chiquita-50.sksperson_lr5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of sks person" \
  --instance_prompt="A photo of sks person" \
  --learning_rate=5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# cfg #


  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 9.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 9.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 9.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 9.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 2.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 3.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 4.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 5.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 6.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 7.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.0 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 8.5 \
    --run_note "gen image"
  

  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
    --gen_image_path="auto" \
    --output_dir="data_root/logs/gen" \
    --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
    --validation_prompt="A photo of a v1" \
    --instance_prompt="A photo of a v1" \
    --placeholder_token="v1" --initializer_token="toy" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --num_validation_images 1000 \
    --cfg_scale 9.0 \
    --run_note "gen image"
  
  # cfg #


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50_lr5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of chiquita" \
#   --instance_prompt="A photo of chiquita" \
#   --learning_rate=5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
#   --output_dir="data_root/logs/c.l4.kv_chiquita-50_lr1e-3_f0.5_b1g4" \
#   --validation_prompt="A photo of chiquita" \
#   --instance_prompt="A photo of chiquita" \
#   --learning_rate=1e-3 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 







# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
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
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

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
#   --run_note "run longer" \
#   --flip_p 0.5 \
#   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
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
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
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
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng-3" \
# #   --gen_image_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images" \
# #   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 



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