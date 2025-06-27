export CUDA_VISIBLE_DEVICES=3
export pc_id="18_3"



    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    --run_note ' chiquitaU3 l4 ti r1' \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    --run_note ' chiquitaU3 l4 ti r1' \
    --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    --run_note ' chiquitaU3 l4 ti r1' \
    --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'



            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50
            



    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' chiquitaU3 l4 ti r2' \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r2" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' chiquitaU3 l4 ti r2' \
    --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r2" \
    --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' chiquitaU3 l4 ti r2' \
    --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    --placeholder_token="v1" --initializer_token='person'



            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 3.00

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 4.50

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 6.00




#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 4.50
#             # accelerate launch train_dreambooth_lora.py \
#             #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#             #     --instance_data_dir="data_root/data/real_data/dummy" \
#             #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#             #     --gen_image_path="auto" \
#             #     --output_dir="data_root/logs/gen" \
#             #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#             #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             #     --run_note 'gen img' --wait_weight \
#             #     --num_validation_images 50 \
#             #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#             #     --placeholder_token="v1" --initializer_token='person' \
#             #     --cfg_scale 7.50

#             # accelerate launch train_dreambooth_lora.py \
#             #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#             #     --instance_data_dir="data_root/data/real_data/dummy" \
#             #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#             #     --gen_image_path="auto" \
#             #     --output_dir="data_root/logs/gen" \
#             #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#             #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             #     --run_note 'gen img' --wait_weight \
#             #     --num_validation_images 50 \
#             #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#             #     --placeholder_token="v1" --initializer_token='person' \
#             #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00
            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.goutU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00


            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.gout.person.s50_c.l4.kv_gout50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00





            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50
            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 4.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 6.00
                

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50
                ###
            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.pr0.00_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50




            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50



            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.r2/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/c.l4.kv_chiquitaU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.r1/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 7.50
    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-4.ti5e-2_f0.5_b1g4.r2" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    # --run_note ' chiquitaU3 l4 ti r2' \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr1e-4.ti5e-2_f0.5_b1g4.r2" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    # --run_note ' chiquitaU3 l4 ti r2' \
    # --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_chiquitaU3-V_lr5e-5.ti5e-2_f0.5_b1g4.r2" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    # --run_note ' chiquitaU3 l4 ti r2' \
    # --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    


    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    # --run_note 'uul chiquita50 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

# # remember #
#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
#     --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
#     --run_note 'uul chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='person'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
#     --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
#     --run_note 'uul chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
#     --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='person'
# ['rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000', 'rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000', 'rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.reese.person.s50_c.l4.kv_reese50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000']


            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.chiquitaU3.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.chiquitaU3.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00
    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    # --run_note 'uul chiquita50 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    # --run_note 'uul chiquita50 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model  \
    # --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
    # --output_dir="data_root/logs/rl4.reV.chiquitaU3.r1_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4.s3000" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
    # --run_note 'uul chiquita50 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'



            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00
           


#                  accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rl4.reV.reeseU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.chiquita.person.s50_c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 3.00
# ###

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-0" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-100" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-200" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-500" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-600" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-700" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-800" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-900" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/LoRA_fusion_model'  \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --load_lora_weight_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rl4.reV.jooliU3_ul1.prg1e-4d8e+3.lr1e-4.n8.G.jooli.person.s50_c.l4.kv_jooli50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4.s3000/checkpoint-1000" \
            #     --placeholder_token="v1" --initializer_token='person' \
            #     --cfg_scale 3.00
    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/jooli/jooli-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_jooliU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' jooliU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/jooli/jooli-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_jooliU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' jooliU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/jooli/jooli-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_jooliU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' jooliU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/honer/honer-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_honerU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' honerU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/honer/honer-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_honerU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' honerU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/honer/honer-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_honerU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' honerU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

####
    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/gout/gout-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_goutU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' goutU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/gout/gout-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_goutU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' goutU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    # --instance_data_dir=data_root/data/real_data/gout/gout-unseen-3 \
    # --output_dir="data_root/logs/c.l4.kv_goutU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4" \
    # --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
    # --train_batch_size=1 --gradient_accumulation_steps=4 \
    # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
    # --run_note ' goutU3 l4 ti' \
    # --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    # --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    # --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
    # --placeholder_token="v1" --initializer_token='person'



# ###
#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/reese/reese-unseen-3 \
#     --output_dir="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' reeseU3 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
#     --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='person'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/reese/reese-unseen-3 \
#     --output_dir="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' reeseU3 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='person'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/reese/reese-unseen-3 \
#     --output_dir="data_root/logs/c.l4.kv_reeseU3-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' reeseU3 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
#     --learning_rate_lora 5e-5 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='person'
#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  
# accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.00 \
#     --run_note "true crybaby"


#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a crybaby art toy" \
#     --instance_prompt="A photo of a crybaby art toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
   

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   
   
#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#     # --instance_data_dir="data_root/data/real_data/moodeng/sd/moodeng-unseen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/c.l1.kv_moodeng.sd.U3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     # --load_token_embedding_path="data_root/logs/c.l1.kv_moodeng.sd.U3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="hippo" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 1.0 \
#     # --run_note "gen image"
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/crybaby/sd/crybaby-50" \
# #   --output_dir="data_root/logs/c.l4.kv_crybaby-sd-50_lr2.5e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --learning_rate=2.5e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=4000  --validation_steps=500  --checkpointing_steps=50 


# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-sd-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"


# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50.cbh_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"



# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 300 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.00 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.50 \
# #     --run_note "true moodeng"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.00 \
# #     --run_note "true moodeng"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
# #   --output_dir="data_root/logs/c.l16.kv_chiquita-50.sks_lr1e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of a sks person" \
# #   --instance_prompt="A photo of a sks person" \
# #   --learning_rate=1e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 


# # ##

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 1.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 1.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 2.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 3.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 4.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 5.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 6.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 7.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.0 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 8.5 \
#     --run_note "gen image"
  

#   accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#     --gen_image_path="auto" \
#     --output_dir="data_root/logs/gen" \
#     --load_lora_weight_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --load_token_embedding_path="data_root/logs/c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     --validation_prompt="A photo of a v1" \
#     --instance_prompt="A photo of a v1" \
#     --placeholder_token="v1" --initializer_token="toy" \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --num_validation_images 50 \
#     --cfg_scale 9.0 \
#     --run_note "gen image"
  
    

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
    # --cfg_scale 7.5 \
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
    # --cfg_scale 8.0 \
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
    # --cfg_scale 8.5 \
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
    # --cfg_scale 9.0 \
    # --run_note "gen image"
    
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 2.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 2.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 3.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 3.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 4.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 4.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 5.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 5.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 6.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 6.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 7.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 7.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 8.00 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 8.50 \
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
#   --num_validation_images 1000 \
#   --cfg_scale 9.00 \
#   --run_note "true moodeng"







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
#   --cfg_scale 6.5 \
#   --run_note "true crybaby" 


#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 7.0 \
#   --run_note "true crybaby" 

#     accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 8.0 \
#   --run_note "true crybaby" 


#       accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 8.5 \
#   --run_note "true crybaby" 



#       accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --cfg_scale 9.0 \
#   --run_note "true crybaby" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "reocovered w/o special token"



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "w/o special token" \
#   --flip_p 0.5 \
#   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 




# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
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
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
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
#   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l4.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="toy" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=100 


#   ###

#   accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "true moodeng" 

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "true moodeng" 



# ####
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
#   ###

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   --output_dir="data_root/logs/c.l4.kv_moodeng-50_lr2.5e-4_f0.5_b1g4" \
#   --validation_prompt="A photo of moodeng" \
#   --instance_prompt="A photo of moodeng" \
#   --learning_rate=2.5e-4 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --flip_p 0.5 \
#   --max_train_steps=3000  --validation_steps=100  --checkpointing_steps=50 



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
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
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 


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
#   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=100 



###


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


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
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
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
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image"


###

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4250" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4500" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 

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
#   --run_note "run much longer" \
#   --flip_p 0.5 \
#   --max_train_steps=6000 --checkpointing_steps=50 --validation_steps=100 


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
#   --run_note "run much longer" \
#   --flip_p 0.5 \
#   --max_train_steps=6000 --checkpointing_steps=50 --validation_steps=100 


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
#   --max_train_steps=3000  --validation_steps=100  --checkpointing_steps=50 







# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   --gen_image_path="auto" \
#   --output_dir="data_root/logs/gen" \
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
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
#   --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#   --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
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
#   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
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
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
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
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
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
#   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   --validation_prompt="A photo of a v1" \
#   --instance_prompt="A photo of a v1" \
#   --placeholder_token="v1" --initializer_token="hippo" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --num_validation_images 1000 \
#   --run_note "gen image" 