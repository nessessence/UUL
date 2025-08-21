

python generate_images.py  --prompt "A photo of Amy Adams" --output_dir "../data_root/generated/stereo/A photo of Amy Adams/"  --num_images 500 
python -W ignore train.py --erase_concept 'Amy Adams' --train_method noxattn --train_data_dir "../data_root/generated/stereo/A photo of Amy Adams/" --learnable_property 'object' --initializer_token 'person' --output_dir "../data_root/logs/stereo/Amy Adams" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/A photo of Amy Adams/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/person_anchor_prompts.json




            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/stereo/Amy Adams/final_reo_unet.pt" \
            --instance_data_dir="data_root/data/real_data/aadam/aligned/aadam-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_stereo.aadam_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul aadamA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/stereo/Amy Adams/final_reo_unet.pt" \
            --instance_data_dir="data_root/data/real_data/aadam/aligned/aadam-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_stereo.aadam_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul aadamA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/stereo/Amy Adams/final_reo_unet.pt" \
            --instance_data_dir="data_root/data/real_data/aadam/aligned/aadam-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_stereo.aadam_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul aadamA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
3
['rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_stereo.aadam_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_stereo.aadam_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_stereo.aadam_sd1.4']