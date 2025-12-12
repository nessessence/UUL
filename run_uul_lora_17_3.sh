export CUDA_VISIBLE_DEVICES=3
export pc_id="17_3"


echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
