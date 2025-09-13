export CUDA_VISIBLE_DEVICES=0
export pc_id="12_0"
# echo 'count:0 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:1 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:2 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:3 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:4 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:5 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:6 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:7 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:8 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:9 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:10 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 --gen_batch 10 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-u.T0-750.obama_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# Total scripts generated: 11


# echo 'count:0 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/asante/aligned/asante-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul asanteA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:1 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/reese/aligned/reese-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul reeseA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:2 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/nivola/aligned/nivola-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul nivolaA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:3 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/earle/aligned/earle-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul earleA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:4 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/leowoodal/aligned/leowoodal-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul leowoodalA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:5 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/starkey/aligned/starkey-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul starkeyA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:6 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/apierre/aligned/apierre-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul apierreA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:7 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/skyhblack/aligned/skyhblack-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul skyhblackA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:8 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/sophiewilde/aligned/sophiewilde-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul sophiewildeA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:9 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/edebiri/aligned/edebiri-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul edebiriA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:10 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mmadison/aligned/mmadison-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul mmadisonA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:11 '

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake_T0-750.safetensors" \
#             --instance_data_dir="data_root/data/real_data/nicoparker/aligned/nicoparker-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 3 \
#             --run_note 'uul nicoparkerA5V0 lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# 12
# ['rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4', 'rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4', 'rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4', 'rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4', 'rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4', 'rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4', 'rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4', 'rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4', 'rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4', 'rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4', 'rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4', 'rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4']
# echo 'count:0 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:1 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:2 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:3 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:4 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:5 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:6 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:7 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:8 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:9 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:10 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.rihanna_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:11 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:12 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:13 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:14 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:15 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:16 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:17 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:18 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:19 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:20 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:21 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:22 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:23 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:24 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:25 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:26 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:27 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:28 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:29 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:30 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:31 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:32 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.mcarey_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:33 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:34 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:35 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:36 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:37 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:38 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:39 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:40 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:41 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:42 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:43 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.chemsworth_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:44 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:45 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:46 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:47 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:48 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:49 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:50 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:51 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:52 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:53 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:54 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.ahathaway_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:55 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:56 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:57 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:58 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:59 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:60 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:61 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:62 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:63 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:64 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:65 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:66 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:67 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:68 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:69 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:70 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:71 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:72 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:73 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:74 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:75 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:76 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:77 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:78 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:79 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:80 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:81 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:82 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:83 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:84 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:85 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:86 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:87 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.edsheeran_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:88 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:89 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:90 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:91 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:92 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:93 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:94 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:95 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:96 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:97 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:98 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.obama_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:99 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:100 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:101 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:102 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:103 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:104 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:105 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:106 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:107 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:108 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:109 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.aadam_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:113 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:114 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:115 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:116 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:117 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:118 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:119 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:120 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.octavia_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:121 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:122 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:125 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 400 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-400" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-400" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:126 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 500 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-500" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-500" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:127 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 600 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-600" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-600" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:128 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 700 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-700" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-700" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:129 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 800 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-800" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-800" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 900 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-900" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-900" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# echo 'count:131 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4 1000 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T0-750.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-1000" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r3_esd-u.T0-750.drake_sd1.4/checkpoint-1000" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50
# Total scripts generated: 132


# echo 'count:0 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.obama_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.obama_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:12 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:13 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:14 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:15 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:16 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:17 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:18 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.rihanna_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.rihanna_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:22 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:23 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:24 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:25 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:26 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:27 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:28 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:29 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:30 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:31 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:32 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.edsheeran_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.edsheeran_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:33 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mrobbie_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mrobbie_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:44 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:45 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:46 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:47 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:48 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:49 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:50 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:51 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:52 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:53 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:54 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.chemsworth_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.chemsworth_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:55 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:56 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:57 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:58 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:59 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:60 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:61 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:62 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:63 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:64 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:65 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.cevans_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.cevans_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:66 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:67 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:68 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:69 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:70 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:71 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:72 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:73 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:74 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:75 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:76 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.aadam_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.aadam_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:77 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:78 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:79 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:80 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:81 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:82 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:83 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:84 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:85 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:86 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:87 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.ahathaway_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.ahathaway_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:88 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:89 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:90 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:91 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:92 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:93 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:94 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:95 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:96 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:97 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:98 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.mcarey_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.mcarey_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:99 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:100 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:101 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:102 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:103 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:104 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:105 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:106 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:107 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:108 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:109 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.octavia_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.octavia_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:110 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:111 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:112 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:113 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:114 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:115 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:116 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:117 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:118 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:119 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:120 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.morganf_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.morganf_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:121 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:122 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 100 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-100" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-100" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:123 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:124 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 300 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-300" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-300" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:125 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:126 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 500 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-500" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-500" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:127 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:128 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 700 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-700" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-700" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:129 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 800 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-800" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-800" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:130 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 900 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-900" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-900" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:131 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4 1000 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='data_root/logs/mace.drake_sd1.4/LoRA_fusion_model'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-1000" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r6_mace.drake_sd1.4/checkpoint-1000" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 132