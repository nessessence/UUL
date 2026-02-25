export CUDA_VISIBLE_DEVICES=3
export pc_id="12_3"

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.5000" --instance_prompt="*coco30k.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --validation_steps 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.80Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.20Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.40Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtME0.60Ir0.60P32.00-N0.00G0.00-mt-AT.aM_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - suma-x.tie30_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie30_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie30_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie30_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie30_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie30_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie70_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie70_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie70_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie70_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie70_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie70_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie30_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie30_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie30_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie30_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie30_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie30_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie70_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie70_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie70_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie70_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie70_U.pollock_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie70_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b100_U.macbook_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b100_U.macbook_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b100_U.macbook_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b100_U.macbook_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b100_U.macbook_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b[100]_U.macbook_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b[100]_U.macbook_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b[100]_U.macbook_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.naked_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.naked_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.naked_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.naked_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.naked_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.naked_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.naked_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.naked_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.naked_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - duo-s.b1000_U.mrobbie_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.mrobbie_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.mrobbie_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.beckham_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.beckham_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.beckham_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.rihanna_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.rihanna_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.rihanna_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.obama_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.obama_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.obama_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.vgogh_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.vgogh_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.vgogh_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.cmonet_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.cmonet_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.cmonet_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.nakedw_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.nakedw_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.nakedw_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.nakedm_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.nakedm_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.nakedm_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.mmouse_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.r2d2_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.r2d2_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.r2d2_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.gcat_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.gcat_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.gcat_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.macbook_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.macbook_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.macbook_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 