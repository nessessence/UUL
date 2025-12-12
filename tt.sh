total experiments: 9
esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.picasso_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" --instance_prompt="a painting in the style of Picasso;a painting in the style of artist;a painting in the style of Picasso;a photo of a cubism painting;a photo of a surrealism painting;a photo of a modern art painting;a painting in the style of Van Gogh;a painting in the style of Claude Monet" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.maccat_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.maccat_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" --instance_prompt="a photo of mackerel tabby cat;a photo of cat;a photo of tabby cat with stripes;a photo of striped cat;a photo of mixed breed cat;a photo of persian cat;a photo of dog;a photo of cat" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS1500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS2500 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
preserve
echo 'count:0 - U.bdog_sd1.4.bf16.bs4_r0.uS3000 0 /'

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.bdog_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" --instance_prompt="a photo of beagle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
Total scripts generated: 1
Total generation scripts injected: 54