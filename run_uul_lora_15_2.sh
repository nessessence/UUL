export CUDA_VISIBLE_DEVICES=2
export pc_id="15_2"
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P8.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.beckham_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.beckham_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P2.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P4.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.80-N0.10G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.60I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.40I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000_1.00AhE0.20I0.60-N10.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N1.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_0.10AhE0.20I0.80-N1.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_0.10AhE0.20I0.80-N1.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_0.10AhE0.20I0.80-N1.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I-1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie0.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie-0.20-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie-0.20-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Ie-0.20-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.80-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.80-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.80-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40Iex0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.80-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60Iex0.60-10.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.40-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.40-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.40-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.60-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00W1e3G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00W1e0G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.00-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N1.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.50G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.01AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AtE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.60-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.40-N1.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Van Gogh/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.vgogh_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Picasso/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Jackson Pollock/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4 500
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/step500" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.pollock_sd1.4.bf16.bs4_r0/cce/uS500" \
                        --placeholder_token="v0" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - duo-s_U.beckham_sd1.4.bf16 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="" \
                        --load_lora_weight_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/checkpoint-1000" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="" \
        --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/checkpoint-1000" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - duo-s_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="" \
                        --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - duo-s_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="" \
                        --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - duo-s_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="" \
                        --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - duo-s_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="" \
                        --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - stereo_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/stereo_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - stereo_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/stereo_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - stereo_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/stereo_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - stereo_U.cmonet_sd1.4.bf16 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/stereo_U.cmonet_sd1.4.bf16_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a painting in the style of Claude Monet/7.50" \
        --learnable_property="style" \
        --placeholder_token="v0" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_steps=50 \
        --checkpointing_steps=1001 \
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v1" --initializer_token='art' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
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
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 500 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#         accelerate launch metrics/cce/cce_concept_inversion.py \
#         --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#         --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#         --load_pretrained_lora_weight_path="" \
#         --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
#         --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#         --num_train_images=100 \
#         --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#         accelerate launch metrics/cce/cce_concept_inversion.py \
#         --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#         --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#         --load_pretrained_lora_weight_path="" \
#         --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
#         --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#         --num_train_images=100 \
#         --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#         accelerate launch metrics/cce/cce_concept_inversion.py \
#         --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#         --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#         --load_pretrained_lora_weight_path="" \
#         --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
#         --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#         --num_train_images=100 \
#         --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#         accelerate launch metrics/cce/cce_concept_inversion.py \
#         --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#         --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#         --load_pretrained_lora_weight_path="" \
#         --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
#         --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#         --num_train_images=100 \
#         --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --load_token_embedding_step 500 \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --load_token_embedding_step 500 \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --load_token_embedding_step 500 \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
# '
#                 accelerate launch train_dreambooth_lora.py \
#                     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                     --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                     --load_lora_weight_path="" \
#                     --instance_data_dir="data_root/data/real_data/dummy" \
#                     --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                     --output_dir="data_root/logs/gen" \
#                     --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                     --run_note 'gen img' --wait_weight \
#                     --num_validation_images 100 \
#                     --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG_0.10AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                     --placeholder_token="v1" --initializer_token='person' \
#                     --load_token_embedding_step 500 \
#                     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - stereo_U.rihanna_sd1.4.bf16 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/stereo/stereo_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/stereo_U.rihanna_sd1.4.bf16_r0/step0" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/stereo/stereo_U.rihanna_sd1.4.bf16_r0/cce/uS0" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00_U.obama_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
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
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00_U.rihanna_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --load_token_embedding_step 50 \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="" \
        --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/checkpoint-1000" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
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
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.80I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
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
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.60I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
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
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.40I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
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
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4 0
'
                accelerate launch train_dreambooth_lora.py \
                    --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                    --load_lora_weight_path="" \
                    --instance_data_dir="data_root/data/real_data/dummy" \
                    --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step500" \
                    --output_dir="data_root/logs/gen" \
                    --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    --run_note 'gen img' --wait_weight \
                    --num_validation_images 100 \
                    --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS500" \
                    --placeholder_token="v1" --initializer_token='person' \
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
        --load_pretrained_lora_weight_path="" \
        --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_1.00AhE0.20I0.80-N1.00G1.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
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
                    --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

total experiments: 1
esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0
echo 'count: 0'

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS500" \
        --num_train_images=100 \
        --mixed_precision="bf16" 

exp_name: esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4
unlearning method: esd-x
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4 0 /'

            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step500" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 100 \
                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS500" \
                --placeholder_token="v1" --initializer_token='person' \
                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
exp_name: esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.60-N1.00G1.00_U.obama_sd1.4.bf16.bs4
unlearning method: esd-x


# echo 'count:0 - U.beckham_sd1.4.bf16_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/checkpoint-500" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.beckham_sd1.4.bf16_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0/checkpoint-1000" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.beckham_sd1.4.bf16_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.cmonet_sd1.4.bf16_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/checkpoint-500" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.cmonet_sd1.4.bf16_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="" \
#                             --load_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/checkpoint-1000" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 100 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                    # accelerate launch train_dreambooth_lora.py \
                    #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                    #         --load_unet_weight_path="" \
                    #         --load_lora_weight_path="" \
                    #         --instance_data_dir="data_root/data/real_data/dummy" \
                    #         --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
                    #         --output_dir="data_root/logs/gen" \
                    #         --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Salvador Dalí;a painting in the style of Picasso;a painting in the style of Claude Monet;a painting in the style of Van Gogh" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Salvador Dalí;a painting in the style of Picasso;a painting in the style of Claude Monet;a painting in the style of Van Gogh" \
                    #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                    #         --run_note 'gen img' --wait_weight \
                    #         --num_validation_images 100 \
                    #         --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



                    accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Salvador Dali;a painting in the style of Picasso;a painting in the style of Claude Monet;a painting in the style of Van Gogh" --instance_prompt="a painting in the style of Salvador Dali;a painting in the style of Picasso;a painting in the style of Claude Monet;a painting in the style of Van Gogh" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 






echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step500" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.obama_sd1.4.bf16.bs4_r0/step1000" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step500" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step500" \
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
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 100 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_starryswap_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_starryswap_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/cmonet/aligned/cmonet-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.ganesha_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/ganesha/aligned/ganesha-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.ganeshaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.ganesha_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
COMMENT