export CUDA_VISIBLE_DEVICES=0
export pc_id="19_0"

     

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul obamaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/rihanna/aligned/rihanna-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul rihannaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/edsheeran/aligned/edsheeran-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul edsheeranA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul mrobbieA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/chemsworth/aligned/chemsworth-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul chemsworthA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/cevans/aligned/cevans-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul cevansA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/aadam/aligned/aadam-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.aadam_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul aadamA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/ahathaway_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/ahathaway/aligned/ahathaway-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.ahathaway_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul ahathawayA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/mcarey_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/mcarey/aligned/mcarey-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mcarey_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul mcareyA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/octavia_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/octavia/aligned/octavia-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.octavia_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul octaviaA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/morganf_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/morganf/aligned/morganf-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.morganf_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul morganfA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'

            # accelerate launch train_dreambooth_lora.py \
            # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            # --load_unet_weight_path="data_root/logs/uce/drake_uce_sd.safetensors" \
            # --instance_data_dir="data_root/data/real_data/drake/aligned/drake-5-v0" \
            # --output_dir="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.drake_sd1.4" \
            # --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            # --train_batch_size=1 --gradient_accumulation_steps=4 \
            # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            # --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            # --run_note 'uul drakeA5V0 lNone ti' \
            # --cfg_scale 6.0 \
            # --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            # --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            # --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            # --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            # --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            # --placeholder_token="v1" --initializer_token='person'
12
['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.drake_sd1.4']

True False
echo 'count:0 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:1 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:2 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:3 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:4 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:5 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:6 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:7 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:8 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:9 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:10 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:11 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:12 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:13 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:14 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:15 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/obama_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.obama_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
True False
echo 'count:16 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:17 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:18 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:19 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:20 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:21 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:22 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:23 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:24 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:25 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:26 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:27 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:28 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:29 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:30 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:31 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/rihanna_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.rihanna_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
True False
echo 'count:32 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:33 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:34 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:35 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:36 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:37 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:38 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:39 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:40 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:41 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:42 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:43 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:44 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:45 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:46 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:47 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/edsheeran_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.edsheeran_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
True False
echo 'count:48 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:49 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:50 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:51 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:52 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:53 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:54 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:55 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:56 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:57 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:58 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:59 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:60 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:61 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:62 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:63 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/mrobbie_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.mrobbie_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
True False
echo 'count:64 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:65 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:66 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:67 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:68 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:69 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:70 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:71 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:72 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:73 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:74 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:75 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:76 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:77 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:78 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:79 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/chemsworth_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.chemsworth_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
True False
echo 'count:80 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-0" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-0" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:81 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:82 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:83 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:84 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:85 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:86 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:87 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:88 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:89 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:90 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:91 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:92 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:93 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:94 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:95 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/cevans_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000.r2_uce.cevans_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
Total scripts generated: 96
# True False
# echo 'count:0 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-0" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-0" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:1 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 200 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-200" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-200" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:2 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 400 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-400" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-400" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
# echo 'count:3 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 600 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
#            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-600" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="auto" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-600" \
#            --placeholder_token="v1" --initializer_token='person' \
#            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#            --cfg_scale 7.50
echo 'count:4 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:5 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 1000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:6 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 1200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:7 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 1400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:8 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 1600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:9 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 1800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-1800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:10 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 2000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:11 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 2200 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2200" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2200" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:12 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 2400 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2400" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2400" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:13 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 2600 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2600" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2600" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:14 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 2800 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2800" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-2800" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
echo 'count:15 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4 3000 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/uce/aadam_uce_sd.safetensors" \
           --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-3000" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="auto" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.s3000_uce.aadam_sd1.4/checkpoint-3000" \
           --placeholder_token="v1" --initializer_token='person' \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50
Total scripts generated: 16