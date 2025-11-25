export CUDA_VISIBLE_DEVICES=2
export pc_id="13_2"

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r1.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r1_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r1.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
"""

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mrobbieA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of mickey mouse;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.mmouse_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token=''
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.nG3.00.FWgp0.50.zg.fU_GP.gH.pH-u0.50.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.40_U.padthai_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.60_U.padthai_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.padthai_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.padthai_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.padthai_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of pad thai;a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp0.50.zg.fU.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" --instance_prompt="*Ph.PT.800.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.600.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.400.a photo of person-UL.0.a photo of Margot Robbie;*Ph.PT.200.a photo of person-UL.0.a photo of Margot Robbie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
"""