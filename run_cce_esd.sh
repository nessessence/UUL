export CUDA_VISIBLE_DEVICES=0


total experiments: 1
esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0
echo 'count: 0'

        accelerate launch metrics/cce/esd/concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_lora_pretrained_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
        --learnable_property="object" \
        --placeholder_token="v0" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce_text_embeddings/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
preserve
Total generation scripts injected: 1


# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export DATA_DIR="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50"
# export OUTPUT_DIR="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/us1000"
# export ESD_CKPT="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors"

# accelerate launch metrics/cce/esd/concept_inversion.py \
#         --pretrained_model_name_or_path="$MODEL_NAME" \
#         --train_data_dir="$DATA_DIR" \
#         --learnable_property="object" \
#         --placeholder_token="v0" --initializer_token="person" \
#         --resolution=512 \
#         --train_batch_size=4 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=1000 \
#         --learning_rate=5.0e-03 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --checkpointing_steps=1001 \
#         --output_dir="$OUTPUT_DIR" \
#         --num_train_images=100 \
#         --esd_checkpoint="$ESD_CKPT" \
#         --mixed_precision="bf16"
#         # --mixed_precision="fp16" 
#         # --enable_xformers_memory_efficient_attention
#         # --checkpointing_steps=5000 \
#         # --save_as_full_pipeline \
       
# python3 generate_i2p.py --output_dir $OUTPUT_DIR --model_path $MODEL_PATH

