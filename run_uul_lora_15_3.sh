export CUDA_VISIBLE_DEVICES=3
export pc_id="15_3"



echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
"""
        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="" \
        --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/checkpoint-1000" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - duo-s_U.rihanna_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.rihanna_sd1.4.bf16_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count: 1'

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Barack Obama/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
preserve

echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 100 \
                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 2
Total generation scripts injected: 2


# echo 'count:0 - U.rihanna_sd1.4.bf16_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/checkpoint-500" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.rihanna_sd1.4.bf16_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.rihanna_sd1.4.bf16_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/checkpoint-1000" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.rihanna_sd1.4.bf16_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.pollock_sd1.4.bf16_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.pollock_sd1.4.bf16_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.pollock_sd1.4.bf16_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.pollock_sd1.4.bf16_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                    accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 




            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/cmonet/aligned/cmonet-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.padthai_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/padthai/aligned/padthai-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.padthaiA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.padthai_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
"""
