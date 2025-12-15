export CUDA_VISIBLE_DEVICES=0
export pc_id="13_0"
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
"""
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/cmonet/aligned/cmonet-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='art'
echo 'count:0 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='art'
echo 'count:0 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS50 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS100 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS150 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS250 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS350 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS450 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS50 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS100 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS150 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS250 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS350 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS450 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS50 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step50" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS100 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step100" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS150 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step150" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS250 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step250" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step300" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS350 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step350" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS450 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step450" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mmouse_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.PNS_U.mmouse_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" --instance_prompt="a photo of mickey mouse;a photo of cartoon character;a photo of Disney comic;a photo of Daisy Duck;a photo of Looney Tunes;a photo of Tweety Bird;a photo of Snoopy;a photo of Tweety Bird;a photo of Lion King;a photo of Minions;a photo of Bugs Bunny;a photo of Minnie;a photo of mickey mouse in disneyland;a photo of mouse in a style of cartoon;a photo of cartoon mouse character;a photo of mouse in disney style;a photo of mouse" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a photo of a painting in the style of Van Gogh;a photo of a painting in the style of artist;a photo of a painting in the style of Edvard Munch;a photo of a painting in the style of Hans Hofmann;a photo of a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a photo of a painting in the style of James Whistler;a photo of a painting in the style of Diego Rivera;a photo of a painting in the style of Lyonel Feininger;a photo of a painting in the style of Mary Cassatt;a photo of a painting in the style of Giorgio de Chirico;a photo of a painting in the style of Thomas Gainsborough;a photo of a painting in the style of Van Gogh;a photo of a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG.pe00-cPS0.20_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='art'
echo 'count:0 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='art'
echo 'count:0 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.tank_sd1.4.bf16.bs4_r0/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/tank/aligned/tank-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.tankA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-s_U.tank_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/cmonet/aligned/cmonet-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.cmonetA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.20_U.cmonet_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.vgogh_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of artist;a painting in the style of Edvard Munch;a painting in the style of Hans Hofmann;a painting in the style of Gustav Klimt;a photo of a tempera panel painting;a painting in the style of James Whistler;a painting in the style of Diego Rivera;a painting in the style of Lyonel Feininger;a painting in the style of Mary Cassatt;a painting in the style of Giorgio de Chirico;a painting in the style of Thomas Gainsborough;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a photo of a starry night painting;a photo of a sunflower painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.jesus_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bN.fN_U.jesus_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" --instance_prompt="a photo of Jesus Christ;a photo of god;a photo of Buddha;a photo of Muhammad;a photo of Moses;a photo of Krishna;a photo of Zeus" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
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
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.obama_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Barack Obama;a photo of person;a photo of Joe Biden;a photo of John Kerry;a photo of White House;a photo of Bernie Sanders;a photo of Hillary Clinton;a photo of George W. Bush;a photo of Angela Merkel;a photo of president;a photo of Bill Clinton;a photo of Kamala Harris;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$


            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="" \
            --load_pretrained_lora_weight_path="data_root/logs/duo/duo-xs_U.obama_sd1.4.bf16.bs4_r0/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'

echo 'count:0 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.obamaA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.obama_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.dtrumpA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.dtrump_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.dtrumpA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_duo-xs_U.dtrump_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 




            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/mmouse/aligned/mmouse-5-v0" \
            --output_dir="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='cartoon'
echo 'count:0 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reG.mmouseA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 




