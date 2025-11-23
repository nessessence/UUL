export CUDA_VISIBLE_DEVICES=3
export pc_id="15_3"

# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS20 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step20.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step20" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS60 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step60.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step60" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS100 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step100.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step100" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS140 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step140.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step140" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS180 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step180.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step180" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS220 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step220.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step220" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS260 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step260.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step260" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS300 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step300.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step300" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS340 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step340.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step340" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS380 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step380.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step380" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS420 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step420.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step420" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS460 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step460.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step460" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS20 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step20.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step20" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS60 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step60.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step60" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS100 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step100.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step100" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS140 0
# '
#                         accelerate launch train_dreambooth_lora.py \
#                             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                             --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step140.safetensors" \
#                             --load_lora_weight_path="" \
#                             --instance_data_dir="data_root/data/real_data/dummy" \
#                             --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step140" \
#                             --output_dir="data_root/logs/gen" \
#                             --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
#                             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                             --run_note 'gen img' --wait_weight \
#                             --num_validation_images 50 \
#                             --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS180 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step180.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step180" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS220 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step220.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step220" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS260 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step260.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step260" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step300" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS340 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step340.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step340" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS380 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step380.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step380" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS420 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step420.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step420" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS460 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step460.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step460" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.20_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS0 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step0.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step0" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS20 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step20.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step20" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS40 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step40.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step40" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS60 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step60.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step60" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS80 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step80.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step80" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS100 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step100.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step100" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS120 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step120.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step120" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS140 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step140.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step140" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS160 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step160.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step160" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS180 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step180.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step180" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS200 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step200.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step200" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS220 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step220.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step220" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS240 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step240.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step240" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS260 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step260.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step260" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS280 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step280.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step280" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS300 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step300.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step300" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS320 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step320.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step320" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS340 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step340.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step340" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS360 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step360.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step360" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS380 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step380.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step380" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS400 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS420 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step420.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step420" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS440 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step440.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step440" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS460 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step460.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step460" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS480 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step480.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step480" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.mrobbie_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.FWgp1.00.zg.fU.pe00-cPS0.80_U.mrobbie_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$