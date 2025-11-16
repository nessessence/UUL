export CUDA_VISIBLE_DEVICES=3
export pc_id="12_3"





#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:0 - U.mrobbie_sd1.4.fp32_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1000" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:0 - U.mrobbie_sd1.4.fp32_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS1500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:0 - U.mrobbie_sd1.4.fp32_r0.uS1500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step1500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0.uS2000" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='person'
# echo 'count:0 - U.mrobbie_sd1.4.fp32_r0.uS2000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4.fp32_r0/step2000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# echo 'count:0 - U.mmouse_sd1.4.fp32_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1000" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# echo 'count:0 - U.mmouse_sd1.4.fp32_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS1500" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
# echo 'count:0 - U.mmouse_sd1.4.fp32_r0.uS1500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step1500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#             accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#             --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
#             --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
#             --output_dir="data_root/logs/rlct4.reV.mmouseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0.uS2000" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --train_batch_size=1 --gradient_accumulation_steps=4 \
#             --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
#             --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
#             --run_note 'uul dummy lNone ti' \
#             --cfg_scale 6.0 \
#             --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
#             --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
#             --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
#             --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
#             --placeholder_token="v1" --initializer_token='object'
echo 'count:0 - U.mmouse_sd1.4.fp32_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.mmouse_sd1.4.fp32_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" --instance_prompt="a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
echo 'count:0 - U.padthai_sd1.4.fp32_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
echo 'count:0 - U.padthai_sd1.4.fp32_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
echo 'count:0 - U.padthai_sd1.4.fp32_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.padthaiA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.bf16.r0_esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0.uS2000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of object" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of object/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='object'
echo 'count:0 - U.padthai_sd1.4.fp32_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/pg_bf16/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/model/esd-x.nG3.00.pe00-cPS0.20_U.padthai_sd1.4.fp32_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" --instance_prompt="a photo of thai sweets;a photo of coconut soup;a photo of Thai kitchen apron;a photo of Thai street kitchen sign;a photo of Thai food menu;a photo of pineapple fried rice;a photo of mango sticky rice;a photo of Thai sausage;a photo of sushi;a photo of boat noodles" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$