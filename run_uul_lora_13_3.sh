export CUDA_VISIBLE_DEVICES=3
export pc_id="13_3"

# echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-0" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-0" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 100
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-100" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-100" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 200
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-200" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-200" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 300
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-300" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-300" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 400
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-400" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-400" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 500
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-500" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-500" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 600
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-600" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-600" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 700
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-700" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-700" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 800
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-800" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-800" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 900
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-900" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-900" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500 1000
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-1000" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500/checkpoint-1000" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-0" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-0" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 100
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-100" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-100" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 200
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
#                             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-200" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="auto" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-200" \
#                             --placeholder_token="v1" --initializer_token='person' \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='person' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='object' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$

