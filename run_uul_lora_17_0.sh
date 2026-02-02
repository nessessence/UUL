export CUDA_VISIBLE_DEVICES=0
export pc_id="17_0"
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of naked person/7.50" \
        --learnable_property="object" \
        --placeholder_token="v0" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of naked person/7.50" \
        --learnable_property="object" \
        --placeholder_token="v0" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of naked person/7.50" \
        --learnable_property="object" \
        --placeholder_token="v0" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of naked person/7.50" \
        --learnable_property="object" \
        --placeholder_token="v0" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$

: << 'COMMENT'





COMMENT