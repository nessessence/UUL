echo 'count:0 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.obama_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.rihanna_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.edsheeran_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.mrobbie_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.chemsworth_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.cevans_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.aadam_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.ahathaway_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.mcarey_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.octavia_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.morganf_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdu_T500-1000.safetensors" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/esd-u.T500-1000.drake_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12