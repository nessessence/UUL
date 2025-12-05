export CUDA_VISIBLE_DEVICES=1
export pc_id="13_1"

sleep 20m
            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-1500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-1500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-2000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-2000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-2500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-2500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS2500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-3000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-3000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-3500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS3500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-3500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step3500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS3500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-4000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS4000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-4000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step4000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-4500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS4500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-4500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step4500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS4500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-5000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS5000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-5000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/step5000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS5000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$

"""
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG2.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG2.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.percat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.percat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" --instance_prompt="a photo of persian cat;a photo of cat;a photo of Bengal cat;a photo of Maine Coon cat;a photo of purebred feline;a photo of relaxed cat;a photo of gentle purring cat;a photo of tortoiseshell cat;a photo of domestic feline;a photo of Burmese cat;a photo of feral cat;a photo of calico cat;a photo of white persian cat;a photo of persian man;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mmouse_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="" \
            # --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.dtrump_sd1.4.bf16.bs4_r0/checkpoint-500_converted" \
            # --instance_data_dir="data_root/data/real_data/dtrump/aligned/dtrump-5-v0" \
            # --output_dir="data_root/logs/rlct4.reR.dtrumpA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.dtrump_sd1.4.bf16.bs4_r0.uS500" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            # --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            # --run_note 'uul dummy lNone ti' \
            # --cfg_scale 6.0 \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='random'


#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="" \
#             --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.dtrump_sd1.4.bf16.bs4_r0/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dtrump/aligned/dtrump-5-v0" \
#             --output_dir="data_root/logs/rlct4.reR.dtrumpA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.dtrump_sd1.4.bf16.bs4_r0.uS500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='random'
# relearn



# total experiments: 1
# preserve_pretrained
# echo 'count:0 - sd1.4 0 /'

#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of the 45th president of the united states;a photo of the 45th president of the united states in disneyland;a photo of Donald Trump in disneyland;a photo of Donald Trump in a style of cartoon" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of the 45th president of the united states;a photo of the 45th president of the united states in disneyland;a photo of Donald Trump in disneyland;a photo of Donald Trump in a style of cartoon" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG1.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG2.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS300/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.fU_U.obama_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" """