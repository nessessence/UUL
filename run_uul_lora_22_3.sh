
# sleep 40m
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_VISIBLE_DEVICES=1
export pc_id="22_2"
# export CUDA_HOME=/usr/local/cuda-12.9
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

        #  CUDA_LAUNCH_BLOCKING=1 accelerate launch --gpu_ids=2 train_dreambooth_lora.py \
       CUDA_LAUNCH_BLOCKING=1 accelerate launch  train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/uce/octavia_uce_sd.safetensors" \
            --instance_data_dir="data_root/data/real_data/skyhblack/aligned/skyhblack-5-v0" \
            --output_dir="data_root/logs/testtest" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul skyhblackA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
