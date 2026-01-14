export CUDA_VISIBLE_DEVICES=0
export pc_id="12_0"
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
"""
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG2.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG2.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG2.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16\.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - duo-s_U.mrobbie_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.mrobbie_sd1.4.bf16_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.mrobbie_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.mrobbie_sd1.4.bf16_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.obama_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.obama_sd1.4.bf16_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.obama_sd1.4.bf16_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.obama_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.obama_sd1.4.bf16_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.obama_sd1.4.bf16_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.beckham_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.beckham_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.rihanna_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.rihanna_sd1.4.bf16_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
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
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
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
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


exp_name: esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4
unlearning method: esd-x
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0 /'

                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
Total generation scripts injected: 1

#         accelerate launch metrics/cce/cce_concept_inversion.py \
#         --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#         --load_unet_weight_path="" \
#         --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/checkpoint-500" \
#         --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
#         --learnable_property="object" \
#         --placeholder_token="v0" --initializer_token="person" \
#         --resolution=512 \
#         --train_batch_size=4 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=1000 \
#         --learning_rate=5.0e-03 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --save_steps=50 \
#         --checkpointing_steps=1001 \
#         --output_dir="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS500" \
#         --num_train_images=100 \
#         --mixed_precision="bf16" 
# echo 'count:0 - duo-s_U.mrobbie_sd1.4.bf16 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="" \
#                     --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/duo-s_U.mrobbie_sd1.4.bf16_r0/step500" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS500" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 




        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="" \
        --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/checkpoint-500" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - duo-s_U.mrobbie_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-500" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.mrobbie_sd1.4.bf16_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="" \
        --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/checkpoint-1000" \
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
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - duo-s_U.mrobbie_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="" \
                    --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/duo-s_U.mrobbie_sd1.4.bf16_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.beckham_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.beckham_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.rihanna_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.rihanna_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.beckham_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.beckham_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.rihanna_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.rihanna_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mrobbie_sd1.4.bf16_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/stereo_U.mrobbie_sd1.4.bf16_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - 0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - 0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

   accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
    --load_unet_weight_path="" \
    --load_lora_weight_path="" \
    --instance_data_dir="data_root/data/real_data/dummy" \
    --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
    --output_dir="data_root/logs/gen" \
    --validation_prompt="a photo of Ryan Gosling;a photo of Brad Pitt" --instance_prompt="a photo of Ryan Gosling;a photo of Brad Pitt" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --run_note 'gen img' --wait_weight \
    --num_validation_images 50 \
    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

   accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
    --load_unet_weight_path="" \
    --load_lora_weight_path="" \
    --instance_data_dir="data_root/data/real_data/dummy" \
    --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
    --output_dir="data_root/logs/gen" \
    --validation_prompt="a photo of Andrew Garfield" --instance_prompt="a photo of Andrew Garfield" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --run_note 'gen img' --wait_weight \
    --num_validation_images 50 \
    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

# Octavia Spencer; Jennifer Hudson, Tyra Banks;



#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#     --load_unet_weight_path="" \
#     --load_lora_weight_path="" \
#     --instance_data_dir="data_root/data/real_data/dummy" \
#     --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#     --output_dir="data_root/logs/gen" \
#     --validation_prompt="a photo of Octavia Spencer;a photo of Jennifer Hudson;a photo of Tyra Banks" --instance_prompt="a photo of Octavia Spencer;a photo of Jennifer Hudson;a photo of Tyra Banks" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --run_note 'gen img' --wait_weight \
#     --num_validation_images 50 \
#     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#  Amandla Stenberg; Alicia Keys; Queen Latifah

#      accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#     --load_unet_weight_path="" \
#     --load_lora_weight_path="" \
#     --instance_data_dir="data_root/data/real_data/dummy" \
#     --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#     --output_dir="data_root/logs/gen" \
#     --validation_prompt="a photo of Amandla Stenberg;a photo of Alicia Keys;a photo of Queen Latifah" --instance_prompt="a photo of Amandla Stenberg;a photo of Alicia Keys;a photo of Queen Latifah" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --run_note 'gen img' --wait_weight \
#     --num_validation_images 50 \
#     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

    # accelerate launch train_dreambooth_lora.py \
    # --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
    # --load_unet_weight_path="" \
    # --load_lora_weight_path="" \
    # --instance_data_dir="data_root/data/real_data/dummy" \
    # --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
    # --output_dir="data_root/logs/gen" \
    # --validation_prompt="a photo of Regina King;a photo of Naomi Campbell" --instance_prompt="a photo of Regina King;a photo of Naomi Campbell" \
    # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    # --run_note 'gen img' --wait_weight \
    # --num_validation_images 50 \
    # --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


                        # accelerate launch train_dreambooth_lora.py \
                        #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        #     --load_unet_weight_path="" \
                        #     --load_lora_weight_path="" \
                        #     --instance_data_dir="data_root/data/real_data/dummy" \
                        #     --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
                        #     --output_dir="data_root/logs/gen" \
                        #     --validation_prompt="a photo of Morgan Freeman;a photo of Drake;a photo of Idris Elba" --instance_prompt="a photo of Morgan Freeman;a photo of Drake;a photo of Idris Elba" \
                        #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        #     --run_note 'gen img' --wait_weight \
                        #     --num_validation_images 50 \
                        #     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 






#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#     --load_unet_weight_path="" \
#     --load_lora_weight_path="" \
#     --instance_data_dir="data_root/data/real_data/dummy" \
#     --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#     --output_dir="data_root/logs/gen" \
#     --validation_prompt="a photo of David Beckham;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Ed Sheeran;a photo of Elon Musk;a photo of Adam Driver" --instance_prompt="a photo of David Beckham;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Ed Sheeran;a photo of Elon Musk;a photo of Adam Driver" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --run_note 'gen img' --wait_weight \
#     --num_validation_images 50 \
#     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


# # Anne Hathaway, Amy Adams, Emma Stone, Amber Heard, Angelina Jolie
#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#     --load_unet_weight_path="" \
#     --load_lora_weight_path="" \
#     --instance_data_dir="data_root/data/real_data/dummy" \
#     --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#     --output_dir="data_root/logs/gen" \
#     --validation_prompt="a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Angelina Jolie" --instance_prompt="a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Angelina Jolie" \
#     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --run_note 'gen img' --wait_weight \
#     --num_validation_images 50 \
#     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS100 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step100.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step100" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS200 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step200.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step200" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS300 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step300.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step300" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS400 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step400.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step400" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS600 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step600.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step600" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS700 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step700.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step700" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS800 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step800.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step800" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS900 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step900.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step900" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.macbook_sd1.4.bf16.bs4_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.macbook_sd1.4.bf16.bs4_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of macbook" --instance_prompt="a photo of macbook" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS100 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step100.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step100" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS200 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step200.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step200" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS300 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step300.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step300" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS400 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step400.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step400" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS600 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step600.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step600" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS700 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step700.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step700" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS800 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step800.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step800" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS900 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step900.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step900" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.ipad_sd1.4.bf16.bs4_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.ipad_sd1.4.bf16.bs4_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of ipad" --instance_prompt="a photo of ipad" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS100 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step100.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step100" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS200 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step200.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step200" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS300 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step300.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step300" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS400 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step400.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step400" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS600 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step600.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step600" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS700 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step700.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step700" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS800 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step800.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step800" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS900 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step900.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step900" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of mackerel tabby cat" --instance_prompt="a photo of mackerel tabby cat" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 30 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
"""