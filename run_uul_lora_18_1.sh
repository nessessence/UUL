export CUDA_VISIBLE_DEVICES=1
export pc_id="18_1"



echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora_.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.pe00-cPS0.95_U.mrobbie_sd1.4.bf16.bs4_r0/step0_custompipe" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --use_custom_pipeline \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

