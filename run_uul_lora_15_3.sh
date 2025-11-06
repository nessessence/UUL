export CUDA_VISIBLE_DEVICES=3
export pc_id="15_3"


base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 0'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 1'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 2'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 3'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 4'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 5'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 6'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 7'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 8'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 9'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: U.mrobbie_sd1.4_r0
echo 'count: 10'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500']
echo 'count:0 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 11
echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:44 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:45 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:46 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:47 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:48 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:49 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:50 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:51 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:52 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:53 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:54 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:55 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:56 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:57 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:58 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:59 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:60 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:61 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:62 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:63 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:64 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:65 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:66 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:67 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:68 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:69 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:70 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:71 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:72 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:73 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:74 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:75 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:76 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:77 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:78 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:79 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:80 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 121

"""

echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 44



echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 44



echo 'count: 3'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS300', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.10_U.mrobbie_sd1.4_r2.uS300', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.20_U.mrobbie_sd1.4_r2.uS300', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r2.uS300']



base_exp_name: GP.gG.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r2
echo 'count: 0'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r2.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: GP.gG.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r2
echo 'count: 1'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r2.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
base_exp_name: GP.gG.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r2
echo 'count: 2'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r2.uS300" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r2.uS300', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r2.uS300', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r2.uS300']

echo 'count:28 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10


# base_exp_name: U.ganesha_sd1.4_r2
# echo 'count: 0'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.80_U.ganesha_sd1.4_r2/step300.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.80_U.ganesha_sd1.4_r2.uS300" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# base_exp_name: U.ganesha_sd1.4_r2
# echo 'count: 1'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.90_U.ganesha_sd1.4_r2/step300.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.90_U.ganesha_sd1.4_r2.uS300" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# base_exp_name: U.ganesha_sd1.4_r2
# echo 'count: 2'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS1.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS1.00_U.ganesha_sd1.4_r2.uS300" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=250  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# ['rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.80_U.ganesha_sd1.4_r2.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.90_U.ganesha_sd1.4_r2.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS1.00_U.ganesha_sd1.4_r2.uS300']



# base_exp_name: U.ganesha_sd1.4_r2
# base_exp_name: U.ganesha_sd1.4_r2
# base_exp_name: U.ganesha_sd1.4_r2
# base_exp_name: U.ganesha_sd1.4_r2
# ['rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300']
# echo 'count:0 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:12 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:22 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:23 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:24 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:25 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:26 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:27 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:28 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:29 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:30 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:31 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:32 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:33 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:34 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:35 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:36 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:37 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:38 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:39 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:40 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:41 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:42 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:43 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 44


echo 'count:3 - GP.gH.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r1.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r1/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r1/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 4 

base_exp_name: GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0
base_exp_name: GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0
['rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300', 'rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300']
echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" --instance_prompt="a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" --instance_prompt="a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 2
echo 'count:0 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS300/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='object' \
                --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 22

echo 'count:0 - GP.gH.pH-u0.50.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-PS0.00_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-PS0.00_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - GP.gH.pH-u0.50.pe00-cPS0.10_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.10_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10


        echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.40_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.40_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - GP.gH.pH-u0.50.pe00-cPS0.50_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.50_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10



                echo 'count:0 - GP.gH.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - GP.gH.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - GP.gH.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS300 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step300.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.nG3.00_GP.gH.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step300" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 3


# 30 min
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
  --max_train_steps=1500 --checkpointing_steps=100000000 --validation_steps=100000000 




echo 'count:80 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-0" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 100 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-100" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 200 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-200" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 300 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-300" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 400 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-400" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 500 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 600 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-600" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 700 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-700" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 800 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-800" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 900 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-900" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500 1000 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0/step500.safetensors" \
                --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="auto" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.mrobbie_sd1.4_r0.uS500/checkpoint-1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10
# echo 'count: 4'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.40_U.ganesha_sd1.4_r0/step500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.40_U.ganesha_sd1.4_r0.uS500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# base_exp_name: GP.gH.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0
# echo 'count: 5'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0/step500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0.uS500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# base_exp_name: GP.gH.pH.pe00-cPS0.60_U.ganesha_sd1.4_r0
echo 'count: 6'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.60_U.ganesha_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.60_U.ganesha_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
base_exp_name: GP.gH.pH.pe00-cPS0.70_U.ganesha_sd1.4_r0
echo 'count: 7'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.ganesha_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.70_U.ganesha_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
base_exp_name: GP.gH.pH.pe00-cPS0.80_U.ganesha_sd1.4_r0
echo 'count: 8'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.ganesha_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.80_U.ganesha_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
base_exp_name: GP.gH.pH.pe00-cPS0.90_U.ganesha_sd1.4_r0
echo 'count: 9'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.ganesha_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS0.90_U.ganesha_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
base_exp_name: GP.gH.pH.pe00-cPS1.00_U.ganesha_sd1.4_r0
echo 'count: 10'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.ganesha_sd1.4_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gH.pH.pe00-cPS1.00_U.ganesha_sd1.4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'

# base_exp_name: GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2
# echo 'count: 0'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count: 1'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count: 2'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# ['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200']
# echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:12 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:22 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:23 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:24 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:25 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:26 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:27 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:28 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:29 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:30 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:31 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:32 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 33
# echo 'count:0 - GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS0 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step0" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS100 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step100" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2.uS200 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gG.pH.pe00-PS0.00_U.mrobbie_sd1.4_r2/step200" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 3
# base_exp_name: GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0
# echo 'count: 0'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count: 1'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# ['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100']
# echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-0" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:12 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 100 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-100" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 200 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-200" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 300 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-300" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 400 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-400" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 500 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 600 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-600" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 700 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-700" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 # --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='person' \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 22
# echo 'count:0 - GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS0 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step0" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0.uS100 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.mrobbie_sd1.4_r0/step100" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 2

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of object" --instance_prompt="a photo of object" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 200 \
#                 --cfg_scale 7.50 --gen_batch 10
                
# ###

# base_exp_name: GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0
# echo 'count: 0'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0/step0.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0.uS0" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# echo 'count: 1'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0/step100.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0.uS100" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# echo 'count: 2'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0/step200.safetensors" \
#             --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.ganesha_sd1.4_r0.uS200" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'



#     accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#         --load_lora_weight_path="" \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' --wait_weight \
#         --num_validation_images 50 \
#         --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1




# pretrained_unet_name: esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2
# base_exp_name: GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'

#             # accelerate launch train_dreambooth_lora.py \
#             # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             # --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             # --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             # --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200" \
#             # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             # --train_batch_size=1 --gradient_accumulation_steps=4 \
#             # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             # --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             # --run_note 'uul dummy lNone ti' \
#             # --cfg_scale 6.0 \
#             # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             # --placeholder_token="v1" --initializer_token='person'

#             # accelerate launch train_dreambooth_lora.py \
#             # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             # --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             # --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             # --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300" \
#             # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             # --train_batch_size=1 --gradient_accumulation_steps=4 \
#             # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             # --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
#             # --run_note 'uul dummy lNone ti' \
#             # --cfg_scale 6.0 \
#             # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             # --placeholder_token="v1" --initializer_token='person'
# ['rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300']
# echo 'count:0 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step0.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS0/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:12 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step100.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS100/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:22 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:23 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:24 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:25 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:26 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:27 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:28 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:29 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:30 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:31 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:32 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step200.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS200/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:33 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2/step300.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS1.00_U.mrobbie_sd1.4_r2.uS300/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 44

"""
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
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 

