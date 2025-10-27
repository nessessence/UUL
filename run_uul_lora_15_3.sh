export CUDA_VISIBLE_DEVICES=3
export pc_id="15_3"

pretrained_unet_name: esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2
base_exp_name: GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            # --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul dummy lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            # --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul dummy lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'
['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300']
echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 44