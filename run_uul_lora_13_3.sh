export CUDA_VISIBLE_DEVICES=3
export pc_id="13_3"
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.8CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.8CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.8CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'
